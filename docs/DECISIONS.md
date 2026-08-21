# Decision Log

Every significant decision in this system and the reasoning behind it, written
as I made them.

Most are backed by a measurement against the real corpus rather than a
preference — where that is the case, the number is given. A few I got wrong
first and changed; those are in their own section, because how a design was
revised says more than the design.

---

## 1. Constraints I set before writing code

These shaped everything else, so they come first.

### D-01 — The system must handle any document, clean or messy

The reference documents I started from were short, clean and well-structured. I
decided early not to build for them. Real corpora are 100–200 pages, scanned,
inconsistently formatted, and full of things nobody anticipated — and a system
tuned to a tidy sample fails on contact with the first real one.

**Consequence:** the parser is a router (`parsing/router.py`), not a
suffix-switch. PDF, DOCX, XLSX, PPTX, CSV, TSV, TXT, MD, HTML, JSON and images
are all ingestable; a PDF with no extractable text is detected by character
density and sent to Azure Document Intelligence for OCR. Unsupported formats
raise a typed error rather than a bare `ValueError`. A document that cannot be
parsed is recorded as an `IngestError` and the run continues — at five million
documents some fraction will be corrupt or password-protected, and aborting the
batch on one bad file is the wrong failure mode.

### D-02 — The graph must extract real relationships, not just structure

A document-structure graph answers *where is this text*. It does not answer
*what does this policy require of whom*, which is the only reason to have a
graph next to a vector index at all. So the graph uses LLM relationship
extraction, on any document type, clean or not.

I changed my mind to get here — see §3.

### D-03 — Tables never go through a model

A spreadsheet is already relational. Sending it to an LLM is both more
expensive and *less accurate*: a model transcribing a rate table can get a
number wrong where a parser cannot.

**Consequence:** `extraction/tabular.py` derives entities and typed values
deterministically, with `deterministic=True` and `confidence=1.0` on every
edge. Triage routes every table to it. The reference corpus's tables produce **62
entities and 172 relations at zero LLM cost**, and a test asserts zero network
activity by blocking sockets rather than by trusting a comment.

### D-04 — Extraction has to be affordable at 5M documents

At 100–200 pages each that is 500M–1B pages. A naive per-chunk pass is not
expensive, it is impossible, and it is the reason most GraphRAG projects stay
proofs of concept. I treated cost as a design constraint rather than something
to measure afterwards.

**Consequence:** seven cost controls, each measured. Triage removes 35% of
units before a token is spent; units are sections rather than chunks; small
units are packed to a token budget; the system prompt is a fixed ≥1024-token
prefix so Azure prompt caching engages; results are memoized by content hash;
concurrency is bounded with backoff; and the Batch API halves the price for a
bulk backfill. Measured: **2 LLM calls instead of 109, $0.034 for the corpus,
$3.09 per 1000 documents, $0.00 to re-ingest.**

### D-05 — Build the whole capability first, optimise second

A capability that does not exist cannot be improved, only rebuilt. I chose
breadth before polish: every stage of the pipeline exists and runs against
live services, rather than a subset being production-hardened while the rest is
missing. Where something is deliberately shallow it is labelled as such in
[`PRODUCTION.md`](PRODUCTION.md) rather than left to be discovered.

### D-06 — Departments are configuration, not code

One env variable drives what is monitored, ingested, labelled, written to both
stores, and retrievable — so departments and sources can be added or removed
without touching code.

**Consequence:** `rag/departments.py` is the single registry. `DEPARTMENTS`
drives all of it; `DEPARTMENT_SOURCES` optionally decouples a department's name
from its folder; `UNKNOWN_DEPARTMENT_POLICY` decides what happens to anything
else. Adding a sixth department is an env edit plus a folder.
`GET /departments` exposes the list, so the claim is observable rather than
asserted.

### D-07 — Deletes must propagate to both stores

ETL means a document removed from the source is removed everywhere. I assumed
CocoIndex handled this and checked rather than trusting it — it does not. Its
memoization skips unchanged documents but performs no target reconciliation,
and its "N deleted" statistic is mount bookkeeping that never reaches a sink.
The vector store had an explicit delete path; **the graph had none**, so a
removed document left its whole subgraph orphaned forever.

**Consequence:** `Neo4jGraphWriter.delete_document` plus an orphan sweep for
entities that lose their last provenance, called alongside the search sink's
delete. Verified live: ingest → assert in both stores → remove the source file
→ reconcile → assert gone from both.

### D-08 — Retrieval uses both vector and graph, behind a FastAPI service

LangGraph owns orchestration and conversation state; pydantic-ai owns each
typed LLM call. One agent loop, not two stacked on the same decision — that
buys nothing and makes a failure impossible to attribute. Retrieval fuses a
hybrid vector search and a graph traversal through RRF.

### D-09 — The code must read as a logical flow, not a pile of features

Each module owns one decision and is named for it; the pipeline is an explicit
state machine rather than nested conditionals; comments explain *why* rather
than restating the code. The test suite has the same shape as the system.

---

## 2. Engineering decisions

### D-10 — Hybrid search with semantic reranking, not vector-only

Pure vector search cannot find `$5,250` — it blurs into every other dollar
figure in the corpus. BM25 finds it exactly. Azure's semantic ranker is a
cross-encoder over the top 50 fused candidates, a genuinely different judgement
from cosine similarity, and `vector_k` is 50 because fewer candidates leaves
the ranker nothing to rescore.

### D-11 — One shared index with enforced filters, not index-per-department

Cross-department questions stay possible, there is one schema to maintain, and
Azure vector quota is per *partition* — so N indexes buy no capacity.
Index-per-department is documented in [`PRODUCTION.md`](PRODUCTION.md) as the
compliance-grade upgrade, chosen for isolation requirements rather than scale.

### D-12 — Superseded documents stay indexed

The 2025 rate card still governs a contract signed in 2025. Deleting it would
make a legitimate question unanswerable. Versioning is therefore three jobs —
detect, rank, disclose — and the third is what turns a coin flip into a useful
answer: the response says which version it used and from when.

### D-13 — A SQLite document registry for supersession

Supersession is a fact about a *pair* of documents, and CocoIndex processes each
independently, so the per-document path can never resolve it. Re-parsing the
corpus to answer a question about headers is unaffordable; reading the hints
back from Neo4j would make versioning in the vector store silently depend on
the graph being enabled. A small sidecar written during ingest costs nothing and
couples nothing. Write-back is a field merge, so 1536-dimension vectors are
never shipped back to change a boolean.

### D-14 — Adaptive depth rather than full agentic RAG on every query

Running decomposition, parallel retrieval and a corrective round on every
question is how agentic RAG gets slow and expensive for no gain; most enterprise
questions are single-fact lookups. The planner classifies first, and only
multi-part questions pay for decomposition.

### D-15 — Retrieve before asking for clarification

An ambiguous-looking question is a *suspicion*, not a verdict. Clarification
requires competing evidence: two or more passages within 80% of the top score,
from different documents. Asking a user to choose between readings the evidence
has already settled is what makes clarification prompts irritating.

### D-16 — Numeric citation markers, not prose citations

`[3]` either points at a supplied passage or it does not, and that is checkable
at zero cost. A prose citation cannot be checked at all — which is the mechanism
behind "a wrong answer with a valid-looking citation".

### D-17 — Two independent grounding gates

A sufficiency gate *before* generation (a model handed weak evidence and told to
answer will find something to say) and claim-level verification *after* it.
They fail independently, which is the point. A single contradicted claim fails
the whole answer regardless of the ratio.

### D-18 — Citation problems split by severity

An unresolvable marker blocks the answer; a figure missing its own marker is a
warning. Blocking on marker placement would reject correct, well-sourced
answers over punctuation, and the claim-level audit covers those claims anyway.

### D-19 — The query cache is partitioned by scope at the key

The documented production failure for semantic caches is a loose threshold plus
a shared namespace returning one tenant's answer to another. Two callers with
different scopes cannot share an entry even for an identical question, because
they are entitled to different evidence. Refusals are never cached — replaying
"I don't know" hides the day the missing document arrives.

### D-20 — Progress is streamed, answer tokens are not

The answer is only trustworthy after verification. Streaming tokens the verifier
may reject shows the user a claim that is then withdrawn.

### D-21 — The evaluation baseline is real, not a strawman

Vector-only, top 5, no reranking, no query rewriting, raw history handed to the
retriever, no gate between weak retrieval and a confident answer — the pipeline
this repository had before the retrieval work, and what most RAG tutorials
produce. Both sides are measured with the same instruments, including running
the verifier over the baseline's answers, because measuring one side more
strictly than the other would manufacture the result.

### D-22 — Behaviour is a confusion matrix, not an accuracy number

Answering an unanswerable question and refusing an answerable one are different
failures with different remedies. Folding them together would reward
over-refusal, which is the standard way of gaming a hallucination metric.

### D-23 — Postgres was removed rather than left configured

`COCOINDEX_DATABASE_URL`, `PG_SCHEMA_NAME` and `PG_TABLE_NAME` configured
nothing: CocoIndex v1 keeps its state in an embedded LMDB store, and the
pgvector mirror the other two described was never built. A configuration key
nothing reads is a claim the system does not honour. Postgres remains the
documented production home for the SQLite sidecars; it is not pretended to be
in use today.

---

## 3. Decisions I changed

### The graph was structure-only, and that was the wrong call

My first design built a *document-structure* graph — `Document → Section →
Chunk` derived deterministically from parsing — and I justified it on the
grounds that it could not hallucinate. That is true and it is not the point. A
structure graph tells you where text lives; it cannot answer "what does this
policy require of whom", which is the only thing a graph offers that a vector
index does not. I had optimised for a risk instead of for the capability.

The right answer was not to avoid extraction but to make it **auditable**. Every
relation carries `doc_id`, `source_chunk_id`, `section_path`, `page`,
`department`, `confidence` and `evidence_span`; no edge is written without a
resolvable source chunk; and a relation whose quoted evidence span does not
actually appear in the source text is dropped and counted. One was dropped that
way in the corpus run. A model can invent a relationship — provenance is how it
gets caught.

### Clarification fired when the evidence had already decided

The first version asked a clarifying question whenever an ambiguous-looking
query retrieved evidence from more than one department. Measured, that was
wrong: "What is the tuition reimbursement limit?" scored 2.85 on the top
passage against a tail of ~1.8. The answer was never in doubt, and the system
interrogated the user anyway. Clarification now requires genuinely competing
evidence (D-15).

### Two versions of a document are not two readings

Then it over-corrected in a different direction: because version pairs retrieve
together with near-identical scores by construction, every version-sensitive
question started asking for clarification. "What is the current Enterprise
price?" offered a choice between "sales pricing documents" and "finance price
lists" when the only thing that had happened was that both rate cards came
back. Superseded documents are now collapsed onto their successor before rivals
are counted.

### The design specified a rerank stage that the code never had

RRF reads positions only, so two retrievers agreeing on a mediocre chunk
outranked one strong hit a single retriever found — and the cross-encoder score,
the only signal that actually read the query against the passage, was thrown
away. `VendorContract § 3 Payment Terms` ranks **first at 2.95** in raw search
and fell out of the context entirely; the system abstained on a question the
corpus answers in one sentence. Reranking after fusion was in the design
document and never in the code, which is its own lesson.

### A metric that punished the system for being right

Citation accuracy scored 0.0 when there was nothing to cite, so a correct
abstention scored zero while the baseline's wrong-but-cited answer scored one.
On the first full run this made the improved pipeline's citation accuracy come
out *below* the baseline's — a regression that did not exist. Cases with nothing
to cite are now excluded. A metric that inverts on the behaviour it exists to
encourage is worse than no metric, and I would have reported that number.

---

## 4. Defects found by building against the real corpus

None of these were on the plan. Each surfaced because something else was being
built against real documents — which is the argument for not testing solely
against fixtures.

| Defect | Found by | Effect |
|---|---|---|
| Running page footers indexed as body text | Extraction triage | "Northwind Traders — Internal Use Only Page 2" became its own chunk, diluted neighbouring embeddings, and burned context tokens |
| Every DOCX table filed under the document's last heading | Tabular extraction | Hotel-rate and per-diem tables cited as "10 Contact" — right values, wrong citation, which is the worst combination |
| `is_current` never resolved in the search index | Building the searcher | The vector store had no version signal; "what is the Enterprise price?" was a coin flip between $99 and $109 |
| Azure's content filter crashed the pipeline | A prompt-injection test | A correctly-blocked hostile input returned a 500 instead of being classified out of scope |
| `rag.etl.__init__` shadowed the `rag.etl.app` module | Ingest-resilience tests | `from rag.etl import app` silently returned the App object, not the module |
| Live graph tests collided on a fixed probe id | Concurrent test runs | Intermittent failures that vanished when run alone — the worst kind of flake |
| Token floor dropped a real 38-token policy rule | Triage measurement | The account-lockout rule was excluded from the graph; floor lowered from 40 to 30 |
| `token_set_ratio` scores "Enterprise" vs "Enterprise Plus" at 100 | Entity resolution | Would have merged two distinct subscription tiers into one entity |
| A code change does not invalidate the ingest cache | The DOCX fix appearing not to work | Memoization is keyed on document content, so the index kept serving pre-fix chunks until `--full-reprocess` |
| One transient API timeout destroyed a 45-minute benchmark | Re-running the evaluation | No per-case retry; 50 completed cases lost with the 51st |
| `INCLUDED_PATTERNS` restricted to three formats | Auditing the env files | PPTX, CSV, TXT, MD, HTML, JSON and images were never ingested, despite the router handling them |
