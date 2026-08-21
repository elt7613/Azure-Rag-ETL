# Architecture & Problem-Solving

The questions that actually come up when a RAG system is in production —
retrieval quality, latency, scale, security, cost, and the silent wrong answer.
Where this system already implements something, the file is named so the claim
can be checked rather than taken on trust.

---

## 1. Retrieval quality — "5 chunks come back, only one is relevant"

### Debug it before changing it

The first mistake is to start tuning. Four out of five chunks being irrelevant
has at least six distinct causes, and the fixes for them are mutually
exclusive — raising `top_k` helps one and makes another worse. So the first job
is to find out which failure this is.

Every answer this system produces carries a `diagnostics` block naming what was
searched for, how many candidates came back, the reranker score of each, the
sufficiency verdict and its reason. That is deliberate: you cannot debug
retrieval from the answer text.

The decision procedure:

**Is the answer in the index at all?** Query the index directly for a phrase you
know is in the source document. If nothing comes back, this is not a retrieval
problem — it is a parsing or chunking problem, and the remaining five questions
are a waste of time. Two real examples from this build: running page footers
were being indexed as content (`local_parser._detect_page_furniture`), and every
DOCX table was filed under the document's *last* heading, so a hotel-rate
question retrieved the contact section.

**Is it in the index but in the wrong shape?** Look at the chunk that *should*
have matched. If the answer is split across two chunks, the chunk boundary is
wrong — this system chunks on section boundaries rather than a fixed token
window precisely to avoid cutting a rule in half, and never splits a table.

**Is the chunk retrievable but not retrieved?** Compare pure-vector against
hybrid. If BM25 finds it and vector does not, the query turns on an exact token
— a figure, a product name, a tier — and vector search blurs those together.
That is why retrieval here is hybrid rather than vector-only.

**Is it retrieved but ranked below the noise?** Check `@search.rerankerScore`.
The semantic ranker is a cross-encoder reading the query against the passage,
which is a genuinely different judgement from cosine similarity between two
independently-computed vectors.

**Is the query itself the problem?** A follow-up like "what about Standard?"
retrieves nothing on its own, and pasting the whole conversation into the
retriever is worse. Check the condensed `standalone_query` in the diagnostics.

**Is it a filter problem?** Confirm the caller's department scope. A correct
answer excluded by an access filter looks exactly like a retrieval failure.

### What actually fixed it here

| Change | Why it works |
|---|---|
| Hybrid (BM25 + vector, RRF-fused) | Exact tokens and paraphrases fail differently; using both covers both |
| Semantic reranking over 50 candidates | Cross-encoder relevance, not vector proximity |
| Breadcrumb-prefixed `embed_text` | A chunk carries its document, section and version into its own vector, so "PTO accrual" matches the accrual table and not the word "days" elsewhere |
| Section-aware chunking, tables never split | Stops the answer being cut in half |
| Neighbour expansion | When the right document is found but the wrong chunk, `prev_chunk_id`/`next_chunk_id` pull the adjacent one for the cost of a keyed lookup |
| Query decomposition | A comparison retrieves once per side rather than once for a question that no single chunk answers |

### What I would not do

Raise `top_k` and hope. It converts a precision problem into a context-dilution
problem: the irrelevant chunks are still there, now there are more of them, and
the answer prompt reads top-down. It also raises cost on every single query to
paper over a ranking defect.

---

## 2. Latency — "3 seconds became 12 seconds"

### Find the stage before guessing at the cause

A RAG turn is a chain, and a chain that got 4× slower did so somewhere
specific. Every node in this pipeline records its own duration
(`condense_ms`, `plan_ms`, `retrieve_ms`, `answer_ms`, `verify_ms`, `total_ms`),
so the first question — *which stage?* — is answered by reading one response,
not by profiling.

If those timings do not sum to the total, the missing time is queueing,
connection setup, or DNS, and that is a different investigation from anything
inside the pipeline.

### The realistic causes, in the order they actually occur

**Rate limiting is the most common and the most disguised.** Azure OpenAI 429s
with a `Retry-After`, an SDK retries transparently, and the call simply takes
three times as long. Nothing errors. It looks like the model got slower. Check
the 429 rate, the `x-ratelimit-remaining-*` headers, and your TPM/RPM
allocation before anything else. I saw exactly this in this project's own
evaluation runs: measured p50 latency went from ~12s sequential to ~62s at
concurrency 2, and the code had not changed.

**Context growth.** Generation time scales with input tokens. If retrieval
started returning more or larger chunks — a new document with huge tables, a
`top_k` raised months ago, conversation history accumulating unbounded — the
prompt grew and the model is doing more work. Track prompt tokens per request
as a time series; this system records them per step.

**A retriever that got slower, not the model.** Vector index growth without
added partitions, a filter that stopped being selective, or a semantic ranker
falling back. The stage timings separate this immediately.

**A cold or full cache.** A 20–45% cache hit rate silently becoming 0% —
because a deployment changed the key, or the TTL was shortened — quadruples
average latency with no code change and no errors.

**Serial work that should be parallel.** Sub-queries retrieved one after
another instead of concurrently.

**Something downstream of the answer.** Here, verification is an extra model
call on every answered turn. That is a deliberate cost paid for groundedness,
and it is exactly the kind of thing that gets added and forgotten.

### Structural fixes

Adaptive depth (most questions take the cheap path), parallel sub-query
retrieval, prompt caching on a stable prefix, progress streaming so a
multi-second answer feels responsive, and a p95 alert rather than a mean —
latency is heavy-tailed and a mean hides the tail users complain about.

---

## 3. Scale — 10,000 → 5,000,000 documents

Two orders of magnitude changes which things are hard. Retrieval quality barely
changes; **ingestion economics** and **operational recovery** become the whole
problem.

### What breaks first

**Extraction cost, by a wide margin.** 5M documents × 100–200 pages is
~500M–1B pages. Sending every chunk to an LLM for relationship extraction is
the single line item that makes this project infeasible, which is why the
extraction design is built around not doing that. Measured on this corpus:

| | naive per-chunk | this pipeline |
|---|---|---|
| LLM calls | 109 | **2** |
| Cost | — | **$0.034 total, $3.09 per 1000 documents** |
| Re-ingest cost | full | **$0.00** |

That is a 98% call reduction from triage, section-level units and packing, plus
a content-hash memo cache that makes unchanged sections free. Projected to 5M
documents this is roughly $15k of extraction — a budget line rather than a
blocker — and the Azure OpenAI Batch API halves it again for the backfill.

An honest caveat that belongs in any scale estimate: **output tokens are 96% of
that bill.** Six of the seven cost controls are input-side. The levers that
actually matter at scale are triage (don't extract from it), the memo cache
(don't extract from it twice), and batch pricing. Prompt caching saves ~$0.003
of $0.034 here; presenting it as the big win would be misleading.

### Architecture changes

| Concern | At 10k | At 5M |
|---|---|---|
| Ingestion | In-process, incremental | Queue-driven workers (Service Bus + Container Apps/AKS), horizontally scaled, per-document idempotency keys |
| Vector index | One index, filters | Scalar or binary quantization (4×–28× index reduction), `stored:false` on the vector field, partitions added for quota, sharded by department or tenant where isolation is required |
| Graph | Single Neo4j | Clustered, with entity resolution batched offline rather than inline |
| Extraction | Online | Batch API for backfill, online only for the change stream |
| Dedup | Content hash | Content hash plus near-duplicate detection (MinHash) — at this scale boilerplate is a large fraction of the corpus |
| State | SQLite sidecars | Postgres, so workers share it |
| Recovery | Re-run | Checkpointed, resumable, per-document dead-letter queue |

### The thing people forget

**Reprocessing.** At 5M documents you will change the chunker, the embedding
model, or the ontology at least once, and a full re-index is weeks of compute
and a large bill. Design for it now: version the pipeline, store the version on
every chunk, and support running two versions side by side so a re-index is a
gradual migration rather than an outage. The content-hash cache and the
per-document memoization here are the beginning of that; a production system
needs the pipeline version in the cache key too.

---

## 4. Security — HR documents must never reach Engineering

### The principle

Access control has to be a **property of the query**, not of the response. The
tempting design — retrieve broadly, filter what comes back — is wrong in a way
that is invisible in testing: the content has already been read into a process
that was not entitled to it, and from there it reaches logs, traces, error
messages, and the embedding cache. A filter applied after retrieval protects
the screen, not the data.

### As implemented

- **Deny by default.** A request establishing no scope reads nothing. `resolve_scope` returns an empty list, and an empty scope produces an OData filter that matches nothing rather than one that matches everything (`retrieval/searcher.build_filter`). A caller with no scope gets a 403 explaining why, not an empty answer that would send them debugging the corpus.
- **The caller can only narrow, never widen.** The request body may reduce the scope the gateway granted; asking for a department the gateway did not grant returns nothing. Otherwise any client could grant itself HR.
- **Enforced server-side, in the query.** The department filter goes into the Azure AI Search request and into every Cypher query.
- **Values are escaped.** A department name containing a quote cannot break out of the OData literal; there is a test that tries.
- **The cache is partitioned by scope at the key.** Two callers with different scopes cannot share an entry even for an identical question, because they are entitled to different evidence. The documented production failure for semantic caches is exactly this: a loose threshold plus a shared namespace returning one tenant's answer to another.
- **Chunk ids are not capabilities.** `fetch_chunks` re-checks the department even for ids that came from our own graph.

### For a real enterprise deployment

1. **Identity from Entra ID, not from a header the client controls.** The API sits behind a gateway that authenticates the caller and injects group membership. The shape of the check does not change; its trustworthiness does.
2. **Mirror the source ACLs, don't reinvent them.** Azure AI Search supports indexing ADLS Gen2 POSIX ACLs and enforcing them at query time via `x-ms-query-source-authorization`. That keeps SharePoint/ADLS permissions authoritative — the alternative, a permissions field maintained by the pipeline, drifts the moment someone changes a folder permission.
3. **Index-per-department for compliance-grade isolation.** Filters are a correct control; a separate index is a stronger one, because a filter bug leaks and a missing index cannot. Choose it for regulatory isolation, not for scale — vector quota is per partition, so N indexes buy no capacity.
4. **Never send a document into a model a user may not read.** The generation step sees only what the filtered retrieval returned.
5. **Audit the query, not just the answer.** Log who asked, what scope they had, which chunk ids were returned. "Did anyone in Engineering ever retrieve an HR document?" must be answerable from logs.
6. **Treat embeddings as the document.** A vector is reversible enough to be sensitive; the vector store inherits the document's classification.

---

## 5. Cost — "Azure OpenAI spend jumped"

### Find the cause before optimising

Spend has exactly four inputs: **calls × input tokens × output tokens ×
unit price**. Every real cause moves one of them, and the fix depends on which.

Instrument first. This system records per-call token usage bucketed **per
pipeline step** (`USAGE.by_step`), because the useful question is never "what
did this cost" but "what would I remove". That immediately distinguishes:

- **More calls** — traffic growth (fine), a retry loop (not fine), or a feature that added a model call per turn. Verification added one call per answered turn here; that is a deliberate trade, and it is the kind of thing that gets added and forgotten.
- **More input tokens per call** — `top_k` raised, chunks grown, unbounded conversation history, or a duplicated system prompt. Usually the largest and most invisible.
- **More output tokens** — a prompt that started inviting long answers, or extraction returning more per unit. Worth checking early: on this project's extraction path, output is **96%** of the bill, so an input-side optimisation would have been effort spent on 4% of the problem.
- **Price/tier change** — a deployment moved, or a model version changed.

### The optimisation levers, most effective first

| Lever | Applies to |
|---|---|
| **Don't call the model** | Triage 35% of extraction units out before a token is spent; adaptive depth so simple questions skip decomposition and correction |
| **Don't call it twice** | Content-hash memo cache (re-ingest costs $0.00), exact + semantic query cache |
| **Batch what isn't interactive** | Azure OpenAI Batch API: flat 50% off, 24h window, separate quota |
| **Shrink the context** | Rerank then truncate rather than passing everything; dedupe overlapping chunks; condense history instead of appending it |
| **Prompt caching** | A ≥1024-token invariant prefix with only the variable text last. Measured 71% steady-state prefix hit rate here — real, but input-side, so on this workload it is a minor win |
| **Right-size the model** | `gpt-4.1-mini` for extraction, planning, judging; escalate only on low confidence |
| **Cheaper embeddings** | `text-embedding-3-small` at 1536 dims; MRL truncation and quantization cut storage further |
| **Cap the blast radius** | Per-tenant token budgets and alerts on cost-per-query, not just total spend — total spend rises with traffic, and cost-per-query is the number that reveals a regression |

### On repeated queries

Real semantic-cache hit rates are 20–45%, not the 95% vendors quote. It is worth
doing and it is not a strategy. The correctness risk outweighs the saving if the
key is wrong, which is why scope is part of the key here and refusals are never
cached.

---

## 6. Production failure — "occasionally a completely wrong answer with a valid-looking citation"

This is the most important question, because the failure is **silent**. The
system is not erroring. The user cannot tell. Neither can a dashboard of
success rates.

### The debugging methodology

Reproduce first, and reproduce *deterministically*. Without the exact question,
the conversation history, and the caller's scope, you will not see it — and the
retrieval that produced it may no longer be reproducible if the corpus has
changed. **Log enough to replay a turn**: the condensed query, the retrieved
chunk ids with scores, the assembled context, the prompt, and the raw model
output. This system returns all of that in `diagnostics`. If you cannot replay
the turn, everything below is guesswork.

Then walk the chain, and at each stage ask one question: *did the right thing
reach here, and did the right thing leave?* The bug is at the first stage where
the answer is no.

| Stage | Question | Evidence | If it fails here |
|---|---|---|---|
| **Query** | Did the retriever search for what the user meant? | `standalone_query` | Condensation resolved the follow-up to the wrong subject, or rewrote a self-contained question and drifted it |
| **Retrieval** | Was the correct chunk in the candidate set at all? | `queries_run`, `candidates`, chunk ids | Chunking, embedding, or filter problem — not a generation problem |
| **Ranking** | Did the right chunk survive into the context? | reranker scores | A plausible-but-wrong chunk outranked the right one. **This is the classic cause of this exact symptom**: a chunk about the *2025* rate card outranking the 2026 one, or a similar policy from another department |
| **Context** | Did the model see contradictory or stale evidence? | assembled passages | Two versions of a document, both present, no signal about which governs |
| **Prompt** | Was it told what to do when evidence is thin or conflicting? | system prompt | A prompt with no refusal path makes the model improvise; models do not decline readily |
| **LLM** | Did it use the evidence, or its own knowledge? | claim-level verification | A fact that is true in the world and absent from the corpus is the hardest case, because it survives a plausibility check |
| **Citation** | Does the cited passage actually support the claim? | citation check | The specific defect named in the question |

### Why the citation looks valid

Because nothing checked it. A model asked to cite its sources will produce a
well-formed reference to a real document whether or not that document says what
the sentence claims — the citation is generated text like everything else.

Three mechanisms here address that, and they fail independently, which is why
all three exist:

1. **Citations are numeric markers into the supplied passage list**, not prose. `[3]` either points at a passage that was actually provided or it does not, and that is checked mechanically at zero cost. Free-form citations cannot be checked at all.
2. **Claim-level verification.** The answer is decomposed into factual claims and each is checked against the passages: supported, contradicted, or unsupported. A single contradicted claim fails the whole answer regardless of the ratio — an answer containing one wrong figure is wrong, however good the other four sentences are. Verified live: an answer stating "$7,500" where the corpus says "$5,250" is caught.
3. **Version-aware conflict resolution.** When the evidence contains two versions of one document, the current one ranks first and the answer must say which version it used and from when. The superseded document is *not* deleted — the 2025 rate card still governs a contract signed in 2025 — so this is a ranking and disclosure problem, not a filtering one.

### Once it is fixed

Add the failing question to the evaluation set. A regression that is not in the
golden set will happen again, and "we fixed it" without a test is a statement
about the past. The eval here scores hallucination as a first-class metric —
contradicted claims *plus* answers given to unanswerable questions — precisely
so this class of failure has a number attached to it rather than an anecdote.

### The measurement trap

Over-refusal is the standard way to make a hallucination metric look good. A
system that answers nothing hallucinates nothing. That is why behaviour is
scored here as a confusion matrix — `wrongly_answered` and `wrongly_abstained`
counted separately — rather than folded into a single accuracy number that
would reward the cure being worse than the disease.
