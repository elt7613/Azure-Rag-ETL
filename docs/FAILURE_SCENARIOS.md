# The six failure scenarios

Six ways a basic RAG implementation fails, and what this system does about
each: what actually goes wrong, why, what was built, and the evidence that it
works. Every transcript here is from a live run against a real corpus.

Two of these were not theoretical. Scenario 2 failed on the first real attempt
and the debugging is written up rather than hidden, because the interesting
part of a failure scenario document is the failure.

---

## Scenario 1 — Correct document, wrong chunk

### Why it happens

Six causes, and the fixes are mutually exclusive — which is why the debugging
order in [`ARCHITECTURE_QA.md`](ARCHITECTURE_QA.md#1-retrieval-quality--5-chunks-come-back-only-one-is-relevant)
matters more than the fixes. The ones that bit here:

**A chunk with no context in its own vector.** A chunk saying "20 days" is
about nothing. Embedded alone, it matches any question containing a number.

**Fixed-window chunking cutting a rule in half.** The accrual table and the
sentence that qualifies it end up in different chunks, and whichever is
retrieved is incomplete.

**Vector-only search losing exact tokens.** "$5,250" is a token, not a concept.
Embeddings blur it into every other dollar figure in the corpus.

**Content that should never have been indexed.** Running page footers were
being chunked as body text — noise competing with real content for a place in
the context.

### What was built

| Fix | Where |
|---|---|
| Section-aware chunking; tables never split | `enrich/chunker.py` |
| `embed_text` carries a breadcrumb: document, department, version, section | `enrich/chunker.py` |
| `display_text` stays clean, so the citation and the answer are not polluted by the breadcrumb | same |
| Hybrid BM25 + vector, RRF-fused | `retrieval/searcher.py` |
| Semantic reranking over 50 candidates | same |
| Neighbour expansion via `prev_chunk_id` / `next_chunk_id` | `retrieval/fusion.py` |
| Running headers and footers stripped structurally at parse time | `parsing/local_parser.py` |
| Tables captioned with their section title in the stored content | `enrich/chunker.py` |
| Cross-encoder leads the final ordering, after fusion | `retrieval/fusion.py` |

The last two came out of debugging real failures. Table captioning:  the
reranker scores *stored content*, and a bare markdown grid reads as
structureless next to prose. Measured, the Approval Matrix table ranked below
three prose sections **of its own document** for "who approves an expense of
$3,000" — the table holding the literal answer, outranked by the document's
Purpose section.

And reranking after fusion: RRF reads positions only, so two retrievers
agreeing on a mediocre chunk outranked `VendorContract § 3 Payment Terms` —
which the cross-encoder had ranked **first at 2.95** — straight out of the
context, and the system abstained on a question answered in one sentence. The
design named a rerank stage after fusion and the implementation never had one.

### Evidence

> **Q:** How much PTO does an employee with 4 years of service accrue?
> **A:** An employee with 4 years of service accrues 20 days of PTO annually,
> according to the accrual table for 3–5 years of service [1].
> **Cited:** `HR/LeavePolicy.pdf § 2 Types of Leave > 2.1 Annual / Paid Time Off (PTO)`
> **Groundedness:** 1.0 · contradicted claims: 0

The answer is in a table row — "3 – 5 years | 20 days" — and the question says
"4 years", a number that appears nowhere in the document. Vector similarity
alone does not bridge that; the retrieved chunk had to be the right one and the
model had to read the band.

---

## Scenario 2 — Information across multiple sections

### Why it happens

One chunk cannot answer a two-part question, and nothing about a single
embedding of the whole question guarantees both parts are represented in what
comes back.

### What was built

- **Decomposition.** `agents/planner.py` classifies the question and writes one focused sub-query per fact needed, capped at four. The original question is always retrieved on as well, because decomposition can drop nuance and the whole-question embedding sometimes finds the one chunk answering both halves.
- **Parallel retrieval and RRF fusion** across sub-queries and retrievers.
- **A context budget that scales with the decomposition.**

### This one failed first, and why

The first live run of:

> *I am an IC flying 7 hours internationally and the trip will cost $3,000.
> What cabin am I allowed and who approves the expense?*

produced: **"you are allowed to fly Premium Economy [7] … the specific approval
authority for airfare is not stated in the passages provided."**

Half right, and the missing half *was* in the corpus — `finance/ExpensePolicy.pdf
§7 Approval Matrix` says $2,500–$10,000 requires Department VP approval.

Walking the chain found two independent causes:

1. **The context budget did not scale.** The planner correctly produced three queries, retrieval correctly returned 28 candidates — and then the context was truncated to the same flat top-8 used for a single-fact lookup. Under three chunks per part, and the last part was simply not in the context. The budget now grows per sub-query, capped at what the reranker actually scored.
2. **The table was under-ranked.** Even retrieved, the Approval Matrix lost to prose because it was stored as an uncaptioned grid. Fixed by captioning table chunks with their section title.

Both fixes are small, and neither was guessable from the answer text alone —
they came from the per-node diagnostics showing what was retrieved versus what
reached the model.

### Evidence

*(after the fixes)*

> **Q:** I am an IC flying 7 hours internationally and the trip will cost
> $3,000. What cabin am I allowed and who approves the expense?
> **Plan:** `MULTI_PART` → 2 sub-queries · **Context:** 14 chunks (was 8)
> **A:** As an Individual Contributor flying internationally for 7 hours, you
> are allowed to travel in Premium Economy class [1]. For a trip expense of
> $3,000, approval is required from the Department VP according to the approval
> matrix for expenses between $2,500 and $10,000 [10].
> **Cited:** `finance/TravelPolicy.docx § 3 Air Travel` ·
> `finance/ExpensePolicy.pdf § 7 Approval Matrix`
> **Groundedness:** 1.0 · contradicted claims: 0

Both halves answered, each cited to the table that states it — including the
approval matrix that the first attempt could not reach.

---

## Scenario 3 — Similar documents, conflicting information

### Why it happens

`Pricing2025.pdf` and `Pricing2026.pdf` both say "Enterprise", both look
relevant, and they disagree: $99 versus $109. A system with no version signal
answers from whichever ranked higher, which is a coin flip.

### The real gap this exposed

`extract_metadata` parses a document's own `Supersedes:` header — but the
*consequence* lands on a different document. CocoIndex processes each
independently with nothing to link against, so **every chunk in the index
carried `is_current = true`**, including the superseded 2025 card. The vector
store had no version signal at all.

### What was built

Versioning is treated as three separate jobs, and the third is the one that
turns a coin flip into a useful answer:

1. **Detect** — `targets/version_sync.py` resolves supersession across the whole corpus from a document registry written during ingest, and stamps `is_current` / `superseded_by` onto the indexed chunks with a field merge. `retrieval/conflict.py` also detects unlinked version pairs at query time by title stem and effective date.
2. **Rank** — current versions first; `SUPERSEDES` edges in the graph; a `successors_of` graph query that finds the replacing document even when it shares no wording with the question.
3. **Disclose** — the answer prompt is instructed to name the version it used and its effective date.

**The superseded document stays indexed.** It still governs a contract signed
while it was current, and deleting it would make a legitimate question
unanswerable. This is a ranking and disclosure problem, not a filtering one.

### Evidence

> **Q:** What is the current list price per seat for the Enterprise tier?
> **A:** The current list price per seat for the Enterprise tier is **$109**
> per seat per month, according to `sales/Pricing2026.pdf` version 1.0,
> effective 2026-01-01 [1].
> **Version conflict detected:** *"sales/Pricing2026.pdf is the current
> version; sales/Pricing2025.pdf is superseded"*
> **Groundedness:** 1.0

Both rate cards were retrieved. The system picked the current one, named it,
and gave its effective date — without being asked to.

---

## Scenario 4 — Hallucination / missing information

### Why it happens

A model given weak evidence and an instruction to answer will find something to
say. This is not a defect in the model; it is what the prompt asked for.

### How sufficiency is determined

Two independent gates, one before generation and one after, because they fail
in different ways.

**Before — `retrieval/sufficiency.py`.** Three cheap signals, no extra model call:

| Signal | What it catches |
|---|---|
| Best reranker score (0–4) | Nothing retrieved is a close enough match |
| Score concentration | A flat spread means the query matched a topic, not a fact |
| Query-term coverage | The evidence never mentions what was asked about |

Coverage is the one a threshold alone misses: a passage about leave scores well
against a question about severance, and would be answered from confidently and
wrongly.

**After — `agents/verifier.py`.** The drafted answer is decomposed into factual
claims and each is checked against the cited passages: supported, contradicted,
or unsupported. **A single contradicted claim fails the whole answer**
regardless of the ratio — an answer containing one wrong figure is wrong,
however good the other four sentences are.

Plus mechanical citation checks that cost nothing: a marker either points at a
supplied passage or it does not.

### Evidence

> **Q:** What is the refund policy for Enterprise customers?
> **Behaviour:** abstained
> **A:** I don't have enough in the knowledge base to answer that (the
> evidence does not mention refund, enterprise, customers, so it does not
> address what was asked). The closest documents I found were:
> `finance/ExpensePolicy.pdf`, `finance/TravelPolicy.docx`.

The corpus genuinely has no refund policy — it has cancellation *notice*
periods, which is exactly the adjacent-but-wrong material a naive system
answers from. Note the refusal gives its reason and names the nearest
documents; "I don't know" without either is indistinguishable from a broken
system.

Verification catching a fabrication directly:

> **Passage:** "Employees may be reimbursed up to **$5,250** per calendar year…"
> **Candidate answer:** "Tuition reimbursement is capped at **$7,500** per calendar year [1]."
> **Verdict:** 1 contradicted claim → answer rejected.

---

## Scenario 5 — Ambiguous query

### The decision: retrieve first, then decide

Asking for clarification *before* looking is lazy and usually unnecessary. Most
short questions are not actually ambiguous — they only look it — and a system
that interrogates the user before searching is exhausting to use.

So an ambiguity verdict from the planner is a **suspicion**, and the evidence
confirms or refutes it. Clarification requires *competing* evidence: two or
more passages within 80% of the top score, from different documents.

**Measured, this mattered.** "What is the tuition reimbursement limit?" was
classified ambiguous because the evidence spanned HR and finance — but the top
passage scored 2.85 against a tail of ~1.8. The answer was never in doubt, and
the first implementation asked a pointless clarifying question. The planner was
also over-eager: a question whose *subject* is clear and whose *location* is
uncertain is not ambiguous, and asking a user which department to search is
useless to them.

### Evidence

Genuine ambiguity — clarifies, naming the real options:

> **Q:** What is the limit?
> **Behaviour:** clarify · 73 candidates across 5 queries
> **A:** That could mean a few different things in these documents. Which did
> you mean?
> · Expense limit · API call limit · Leave accrual cap · Contract liability cap

Apparent ambiguity — answers:

> **Q:** What is the tuition reimbursement limit?
> **Behaviour:** answered · **$5,250 per calendar year** [1]

Both are in the evaluation set, and the second exists specifically to catch
over-clarification, which is as much a failure as under-clarifying.

---

## Scenario 6 — Conversational context

### Why it happens

"What about Standard?" retrieves nothing on its own. The obvious fix — pass the
conversation to the retriever — is worse: the earlier turns bring their own
vocabulary and the search drifts back toward the first question.

### What was built

`agents/condense.py` produces a standalone retrieval query from
`(history, message)` and sends **only that** to the retrievers. The raw history
still goes to generation, where it belongs.

Two refinements that matter in practice:

- **The retrieval string is the rewrite concatenated with the raw message.** The rewrite supplies the resolved subject for the vector side; the raw text supplies the user's literal tokens for the keyword side. Rewrite-only measurably loses the second.
- **The rewrite is conservative.** The model is told explicitly that rewriting a self-contained question is a mistake, not a no-op — aggressive expansion amplifies rare terms and pulls retrieval off target. A first turn short-circuits with no LLM call at all.

### Evidence

> **Turn 1:** What is the Enterprise plan cancellation policy?
> **Turn 2:** *What about Standard?*
> **Condensed to:** `"Standard plan cancellation policy What about Standard?"`
> **A:** The Standard plan has a standard contract term of 12 months with
> auto-renewal and can be cancelled with 30 days' written notice prior to
> renewal [1].
> **Cited:** `sales/Pricing2026.pdf § 6 Billing Terms` · **Groundedness:** 1.0

The subject ("cancellation policy") came from turn 1; the retrieval query
carries both it and the user's own words. Note it also answered from the
*current* rate card — conversation handling and version resolution compose
without either knowing about the other.

---

## Summary

| Scenario | Behaviour | Mechanism |
|---|---|---|
| 1 · Wrong chunk | Answered, correct citation | Hybrid + rerank + breadcrumb embeddings + captioned tables + neighbour expansion |
| 2 · Across sections | Answered after two fixes | Decomposition + parallel retrieval + scaled context budget + captioned tables |
| 3 · Conflicting versions | Answered from the current version, said so | Corpus-wide supersession + conflict resolution + disclosure |
| 4 · No answer | Abstained with a reason | Sufficiency gate + claim-level verification |
| 5 · Ambiguous | Clarified with real options | Retrieve-then-decide, competing-evidence test |
| 6 · Follow-up | Answered | History condensed to a standalone query |
