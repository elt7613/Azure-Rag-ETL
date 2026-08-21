# Document and relationship extraction

Two problems live here, and they are usually confused:

1. **Document extraction** — turning any file into structured blocks. Solved by
   routing, not by a bigger model.
2. **Relationship extraction** — turning text into a knowledge graph. This is
   where the money is, and it is the reason most GraphRAG projects quietly stay
   proofs of concept.

Every number below is measured against the eleven real corpus documents with
live Azure services. Where a figure is a projection it says so.

> **About the figures in this document.** They are measured, and they are
> measured against `source_data/` — the eleven multi-page policy documents
> that ship with the project — so they reproduce on a clone. Point the
> pipeline at your own corpus and every method here still holds; the
> absolute values will be your corpus's, not these.


---

## Part 1 — Handling any document

The requirement was explicit: the system must handle any document type, clean
or not, and not be tuned to a small set of clean samples.

### The routing table

| Input | Path |
|---|---|
| `.pdf`, extractable text | pdfplumber |
| `.pdf`, character density below threshold | **scanned** → Azure Document Intelligence OCR |
| `.docx` | python-docx, body walked in true XML order |
| `.xlsx` | openpyxl |
| `.pptx` | python-pptx, slide number as page |
| `.csv`, `.tsv` | table blocks, same shape as xlsx |
| `.txt`, `.md` | markdown headings → heading levels |
| `.html`, `.htm` | stdlib parser, `h1..h6` → levels, tables preserved |
| `.json` | flattened key paths; uniform object arrays → tables |
| `.png/.jpg/.tiff/.bmp` | Document Intelligence OCR |
| anything else | Document Intelligence under `auto`, typed error under `local` |

Scanned detection is measured, not guessed: born-digital corpus pages yield
1,300–1,900 extractable characters per page; the scanned fixture yields **0.0**.
There is no ambiguous middle at the configured threshold of 100.

The router also checks magic bytes, so a PDF named `.docx` is still parsed as a
PDF. File extensions lie in real corpora.

### One bad file must not stop the run

Every per-document failure becomes an `IngestError` carrying the doc id and the
stage it failed at (`read`, `parse`, `embed`, `index`, `graph`), is logged and
counted, and the run continues. At five million documents some fraction will be
corrupt, password-protected, or a format nobody anticipated; aborting the batch
on one of them is the wrong failure mode. The retained error list is capped
while the failure *count* stays exact — an unbounded list of failures is itself
a way to fail a large run.

### Two parsing defects the corpus revealed

Both were found by other work running against real documents, which is the
argument for not testing solely against fixtures.

**Running footers were indexed as content.** `Northwind Traders, Inc. —
Internal Use Only Page 2` became its own chunk in one document and polluted
neighbouring chunks in the rest. Now detected structurally — a line whose
digit-masked text repeats in the top or bottom band of at least half the pages —
rather than by matching the company name. Masking digits is what makes it work;
"Page 1" and "Page 2" look unique otherwise. Restricting it to the edge bands is
what makes it safe: a clause that genuinely recurs mid-document, as legal
templates do, is content and stays.

**Every DOCX table was filed under the document's last heading.** python-docx
exposes `paragraphs` and `tables` as separate collections, each ordered only
within itself, so consuming one then the other put every table after every
paragraph. All three TravelPolicy tables — flight class, hotel caps, per-diem —
were attributed to "10 Contact". The *values* were correct and the *citation*
was wrong, which is the worst combination, because it looks right. Fixed by
walking the body's XML children in document order.

---

## Part 2 — Relationship extraction that scales

### The problem, stated numerically

5M documents × 100–200 pages ≈ 500M–1B pages. A naive pipeline sends every
chunk to an LLM. On this corpus that would be **109 calls for 11 documents** —
roughly 10 calls per document, which at 5M documents is 50M LLM calls before
anything else happens. That is the number that makes GraphRAG-style extraction
infeasible, and the entire design here is about not paying it.

### Two layers, deliberately

**Layer A — structure, deterministic, free.** `Department → Document → Section →
Chunk` with `CONTAINS`, `NEXT` and `SUPERSEDES`, derived from parsing. Cannot
hallucinate. Always present.

**Layer B — semantics, LLM-extracted, triaged.** `Entity` nodes and typed
relations against a closed ontology.

My first design stopped at Layer A and justified it on the grounds that a
deterministic graph cannot hallucinate. That was the wrong trade: a structure
graph answers *where is this text*, not *what does this policy require of
whom*, which is the only thing a graph offers that a vector index does not. The
hallucination concern is answered by **provenance**, not by abstinence — see
[`DECISIONS.md`](DECISIONS.md#3-decisions-i-changed).

### Provenance is the answer to "the LLM might invent a relationship"

It might. Every relation therefore carries `doc_id`, `source_chunk_id`,
`section_path`, `page`, `department`, `confidence`, `evidence_span` and
`deterministic`. Two consequences:

- **No edge without a resolvable source chunk.** An edge that cannot be traced is dropped, not written.
- **The evidence span must actually appear in the source text.** A model quoting something that is not there is the clearest hallucination signal available and it is free to check. **One relation was dropped this way** in the corpus run.

That makes a graph-derived answer as auditable as a vector-derived one.

### The ontology, closed on purpose

*Entities:* Policy, Department, Role, Benefit, Plan, Rate, Vendor, System,
Obligation, Condition, Period, Amount, Location, Process, Metric, Document

*Relations:* APPLIES_TO, ELIGIBLE_FOR, REQUIRES, GRANTS, LIMITS, EXCLUDES,
EXCEPTION_TO, DEFINED_IN, REFERENCES, EFFECTIVE_DURING, HAS_VALUE, OWNED_BY,
APPROVED_BY, SUPERSEDES

A closed set keeps the JSON schema small (so the cached prefix stays small), the
output predictable, and the graph queryable — an open vocabulary produces a
graph where the same relationship appears under six different predicate names
and no traversal finds all of them. The schema is assembled at import from the
same tuples the prompt is built from, so a type cannot reach one without the
other.

### The seven cost controls, and what each is worth

| # | Control | Effect on this corpus |
|---|---|---|
| 1 | **Triage** — skip tables, short units, boilerplate, low-signal prose | 38 of 108 units (35%) never cost a token |
| 2 | **Section-level units, not chunks** | A section spanning five chunks costs one extraction, not five |
| 3 | **Packing to a token budget** | 70 surviving units → **2 calls** |
| 4 | **Content-hash memoization** | A second run over the same corpus: **0 calls, $0.00** |
| 5 | **Cache-friendly prompt shape** — 2,263-token invariant prefix | **71% steady-state prompt-cache hit rate** |
| 6 | **Strict structured outputs + bounded concurrency** | No parse-retry loop; 429s backed off rather than failing |
| 7 | **Batch API for backfill** | 50% off — *projected*, see the caveat below |

### Measured results

| | Naive per-chunk | This pipeline |
|---|---|---|
| LLM calls, 11 documents | 109 | **2** |
| Reduction | — | **98.2%** |
| Prompt / completion tokens | — | 10,752 / 20,551 |
| Entities | — | 393 (LLM) + 62 (tabular) |
| Relations | — | 217 (LLM) + 172 (tabular) |
| Relations dropped for unverifiable evidence | — | 1 |
| **Total cost, whole corpus** | — | **$0.034** |
| **Cost per 1000 documents** | — | **$3.09** |
| Second run, unchanged corpus | full | **$0.00** |

A second independent run measured $3.27 per 1000 documents — normal
run-to-run variance, and worth stating rather than quoting the lower figure.

**Projected to 5M documents: roughly $15,000** of extraction. A budget line
rather than a blocker. Treat that as a *method that reproduces*, not a number
that transfers — five million documents of 100–200 pages have a very different
token profile from eleven two-page policies, and the honest step before
committing to a budget is to benchmark on a real sample.

### The uncomfortable finding

**Output tokens are 96% of the extraction bill** — $0.0329 of $0.034. Six of the
seven controls are input-side. The levers that actually matter at scale are
therefore:

1. **Triage** — don't extract from it
2. **The memo cache** — don't extract from it twice
3. **Batch pricing** — pay half for the backfill

Prompt caching is real, measured, and does what it was built to do — but it
saves about **$0.003 of $0.034**. Presenting it as the big win would be
misleading, so it isn't presented that way.

A note on the cache-hit figure: an early corpus run reported 98.8%, but its
requests were byte-identical to a run made minutes earlier, so Azure served
nearly the whole prompt from cache rather than just the fixed prefix. The
representative number is **71.4%**, measured across four calls over four
*different* sections, where only the prefix can hit.

### Tables never reach the model

A deliberate exception to the rule above: a spreadsheet is already
relational, and a model transcribing a rate table can get a number wrong where
a parser cannot. `extraction/tabular.py` derives entities and typed
values deterministically, with `deterministic=True` and `confidence=1.0`. The
corpus's tables produce **62 entities and 172 relations at zero LLM cost**, and
a test asserts zero network activity by blocking sockets rather than by
trusting a comment.

Typed cell parsing handles the real messiness: `$65` → currency, `15%` and
`0.15` → percentage, `5–24 seats` → range, `20 days` → quantity with unit —
while keeping the raw string, so an answer can quote "$61.75" exactly as
written.

### Entity resolution without an LLM per pair

The extractor produces the same thing under many surface forms. Resolving them
with a model call per candidate pair is the same unaffordable pattern the rest
of this design avoids, so:

1. **Normalize** — casefold, strip legal suffixes, collapse whitespace, handle parenthetical acronyms so "Paid Time Off (PTO)" yields both forms.
2. **Block** — compare only within the same `(department, entity_type)` sharing a token or trigram, with a cap on non-discriminating keys.
3. **Score** — `rapidfuzz`, with cached name embeddings breaking ties near the threshold. Below it, leave unmerged and log: an uncertain merge silently fuses two different real things, which is worse than a duplicate node.

**Measured on 2,000 adversarial synthetic entities** (deliberately sharing
filler words like "Policy" and "Plan" across unrelated clusters):
**27,191 candidate pairs compared against an all-pairs baseline of 1,999,000
— 1.36%.**

One trap worth recording: `token_set_ratio` scores **"Enterprise" against
"Enterprise Plus" at 100**, because it treats the shorter name's tokens as a
subset of the longer's. Those are two distinct subscription tiers in this
corpus and merging them would be a serious error. Taking
`min(token_set_ratio, token_sort_ratio)` keeps word-order independence for
genuine synonyms without rewarding pure containment.

### Batch mode — implemented, not exercisable here

`GRAPH_EXTRACT_MODE=batch` builds the JSONL, submits with
`completion_window="24h"`, polls, and applies results — reusing the online
path's prompt, schema and validation so a batch-extracted relation is checked
identically to an online one.

**Status, measured against a real deployment.** A `datazonebatch` deployment
was provisioned and the batch path points at it through its own endpoint, key
and deployment settings (`AZURE_OPENAI_BATCH_*`, each falling back to the online
values when unset — batch usually has to live on a separate resource, because
one Azure OpenAI account holds a single SKU per model).

The earlier `invalid_deployment_type` rejection is gone. Submission now fails on
**`token_limit_exceeded`**: the deployment's enqueued-token quota is **1K**, and
the smallest possible request is **2,297 tokens** — 2,263 of which are the fixed
system prefix that exists to trigger prompt caching. The whole corpus would
need **~11,700 enqueued tokens across 3 packs**.

So: **no job id has been issued yet, `apply()` is covered by recorded payloads
rather than a live round trip, and the 50% saving remains a projection.** The
remaining step is raising the deployment's batch quota in the portal — it is
separate from the online TPM allocation, so raising it costs nothing and does
not compete with interactive traffic.

### What triage actually skips, and one thing it nearly lost

| Outcome | Units | Reason |
|---|---|---|
| Extracted | 70 | |
| Table | 20 | Deterministic path |
| Too short | 10 | Below the token floor |
| Low signal | 8 | No entity-bearing content |

Two thresholds were set by measurement rather than intuition:

- **The token floor was lowered from 40 to 30.** At 40 it dropped `IT/PasswordPolicy §5 Account Lockout Policy` — a genuine rule (lock after 5 failures, for 30 minutes) stated in **38 tokens**. Terse, high-value policy statements are exactly what the graph exists to capture. A test now pins the trade in both directions.
- **The deontic lexicon had to reach past the policy register.** Restricted to modals, it dropped the NDA's two most relation-dense sections, because contract prose states rules with *agrees to* / *is bound by* / *warrants* rather than *must*.

One accepted loss, tested rather than hidden: `NDA §1 Definition of
Confidential Information` scores below the signal floor because it is almost
entirely proper nouns, which carry deliberately low weight. Its graph value is
thin, and raising proper-noun weight to keep it would start keeping page
furniture.
