# Azure services — what is used, and why

Every service here is doing a job that would otherwise have to be built. Where
an alternative was considered and rejected, the reason is stated: "we chose
Azure AI Search" is not an architecture decision, "we chose it because we need
BM25 and vector search fused server-side with a cross-encoder reranker, and
running that ourselves means operating three components" is.

---

## In use

### Azure OpenAI — `gpt-4.1-mini` and `text-embedding-3-small`

**Used for:** relationship extraction, query planning, history condensation,
answer generation, claim-level verification, and as the evaluation judge.

**Why this model.** Every one of those tasks is structured extraction or
classification against supplied evidence — the task class small models are
already good at. `gpt-4.1-mini` at $0.40/$1.60 per million tokens does all of
them, and the whole knowledge graph is extracted from the corpus for $0.034.
A frontier model would multiply the bill without changing the answers, because
the hard part of grounded RAG is retrieval, not generation.

**Why one deployment for everything.** Fewer moving parts, one quota to manage,
and the escalation path — route low-confidence cases to a larger model — stays
available as a change to one function rather than an architecture change.

**Features that earn their place:**
- **Structured Outputs** (`response_format={"type":"json_schema","strict":true}`) — extraction returns schema-valid JSON with no parse-retry loop. Verified working on this deployment.
- **Prompt caching** — automatic above a 1024-token prefix. The extraction system prompt is deliberately built as a 2,263-token invariant prefix with only the unit text varying, and measures a 71% steady-state cache hit rate.
- **Batch API** — flat 50% discount, 24-hour window, separate quota. The right shape for bulk backfill; online mode handles the change stream. **Blocked on quota, not on capability**: a `datazonebatch` deployment now exists and accepts the request shape, but its enqueued-token quota is 1K against a 2,297-token minimum request, so no job has completed. The saving stays a projection until that quota is raised.
- **Content filtering / Responsible AI** — enforced before the model sees a prompt. It rejects prompt-injection attempts with a 400, which the pipeline treats as a platform verdict and classifies out of scope rather than surfacing a 500.

**Embeddings:** `text-embedding-3-small` at 1536 dimensions. Cheap, and it
supports MRL truncation and quantization if index size becomes the constraint.

### Azure AI Search

**Used for:** the chunk index — BM25, vector search, RRF fusion, semantic
reranking, filtering, and faceting.

**Why.** It is the only component that would otherwise be three: a keyword
engine, a vector index, and a cross-encoder reranker, plus the fusion logic
between them. Specifically:

- **Hybrid in one query.** Text and vector run in parallel and merge through Reciprocal Rank Fusion server-side. Doing this client-side means two round trips and hand-rolled fusion.
- **Semantic ranker.** A cross-encoder rescoring the top 50 fused candidates on a 0–4 scale. This is a genuinely different judgement from cosine similarity, and it is the single largest quality lever in the retrieval path.
- **Filterable fields as security.** `department` and `is_current` are filterable, so access control and version resolution execute *in the query*, not on the results.
- **Scale path that does not require rearchitecting.** Scalar/binary quantization, `stored:false`, partitions for vector quota, replicas for QPS.

**Rejected alternatives.** pgvector or a standalone vector DB (Pinecone, Qdrant,
Milvus): all give vector search, none give BM25 + RRF + a managed cross-encoder
reranker in one call, and reranking is where the accuracy is. Building it means
operating a reranker model.

**Note on this deployment.** The service is serverless, which rejects
`list_indexes()`; the code uses `get_index(name)` guarded by try/except instead.
The installed SDK (`azure-search-documents` 12.0.0) does not expose the native
`queryRewrites` parameter or the Knowledge Agent, so query rewriting and
decomposition are done in the pipeline — which was needed anyway for
conversational condensation.

### Azure AI Document Intelligence (`prebuilt-layout`)

**Used for:** scanned PDFs, images, and any format the local parsers do not
cover.

**Why.** OCR with reading order and real table structure — cell spans, not a
flat text dump. Tesseract gives characters; it does not tell you that four
numbers were a row of a rate table, and losing that turns a pricing table into
prose.

**Why it is not used for everything.** Born-digital PDFs and DOCX parse
perfectly well locally, for free and without a network round trip. The router
sends DI the documents that need it, decided by measured character density
rather than by file extension. `DOC_PARSER=azure` forces it everywhere;
`DOC_PARSER=local` runs with no DI resource at all.

### Azure Blob Storage

**Used for:** the document source in `DOC_SOURCE=blob` mode.

**Why.** It is where enterprise documents already are, and its ETags give
cheap change detection without downloading content. The live source polls
ETags and reprocesses only what actually changed. In production, Event Grid
replaces polling entirely.

Its ADLS Gen2 variant is also the recommended path for real document-level
access control — POSIX ACLs indexed and enforced at query time, keeping the
source system's permissions authoritative rather than mirroring them into a
field that drifts.

### Neo4j

**Used for:** the knowledge graph — both the deterministic structure layer and
the extracted entity layer.

**Why a graph at all.** Some questions are traversals, not similarity searches:
*which policies does the CISO have to approve?*, *what supersedes this?*, *what
else does this obligation apply to?* Vector search answers "what text resembles
this question"; that is a different question, and dressing up a traversal as a
similarity search gets an approximate answer to an exact question.

**Why Neo4j specifically.** Cypher expresses variable-length traversal directly,
MERGE gives idempotent writes (so a re-ingest replaces rather than duplicates),
and the model maps cleanly onto Document → Section → Chunk → Entity. Azure
Cosmos DB for Apache Gremlin is the managed alternative and is noted in the
production design; Gremlin is markedly less readable for the path queries this
uses.

### CocoIndex

**Used for:** the incremental ETL dataflow.

**Why.** Content-fingerprinted memoization out of the box: touching a file
without changing its bytes is a no-op, and only genuine changes cost anything.
Plus a live filesystem watcher (inotify) and a clean lifespan for shared
clients.

**What it does not do, verified rather than assumed.** It has no target
reconciliation. Its "N deleted" statistic is mount bookkeeping and never
reaches a sink — so deletion from the vector store and the graph is explicit
code here, not a framework feature. I assumed CocoIndex handled deletes and
checked rather than trusting it; it does not, and the graph half of the delete
path was missing entirely until I looked.

### FastAPI, LangGraph, pydantic-ai

**FastAPI** for the service: async throughout, which matters when a turn fans
out to several concurrent retrievals; Pydantic models shared with the rest of
the codebase; OpenAPI generated for free.

**LangGraph** for orchestration: the pipeline branches, can abstain mid-flight,
and loops once on verification failure. As a graph each decision is a named
edge that appears in the trace; as nested conditionals it is unreadable at
exactly the moment you need to read it.

**pydantic-ai** for each individual LLM call: typed output validated against a
model, retries handled, usage exposed for accounting.

**Why both.** The idiomatic split is LangGraph owning control flow and
pydantic-ai owning single calls. Stacking two agent loops on the same decision
buys nothing and makes a failure impossible to attribute.

---

## Configured but optional

### Azure AI Content Safety

**Not used, and not configured.** Prompt Shields for injection detection and
the groundedness detection API would be the right addition for a regulated
deployment, where an external and auditable safety verdict has value beyond its
accuracy. I did not wire them: it is a separate resource, and the pipeline
already has two independent grounding gates of its own plus Azure OpenAI's
own Responsible AI filtering, which is always on and is what rejects
prompt-injection attempts before the model sees them.

Settings for it were removed rather than left in place, for the same reason as
Postgres above — a configuration key nothing reads is a claim the system does
not honour.

Note that Azure OpenAI's own Responsible AI filtering is *always* on and is not
optional — it is what rejects injection attempts before the model sees them.

### Application Insights

`APPLICATIONINSIGHTS_CONNECTION_STRING` enables OpenTelemetry export. Left
unset, the same counters are available at `GET /stats`, so observability does
not require an Azure resource to develop against.

### Azure PostgreSQL

**Not used, and not configured.** It is the intended production home for the
extraction cache, the document registry and LangGraph conversation checkpoints —
all SQLite sidecars in this build, which is correct for one process and wrong
for several. Settings for it were removed rather than left in place: a
configuration key nothing reads is a claim the system does not honour, and the
next person to see `PG_TABLE_NAME` would reasonably assume chunks were being
mirrored somewhere. See `PRODUCTION.md` for the migration.

---

## Considered and not used

| Service | Why not |
|---|---|
| **Azure AI Foundry prompt flow** | The pipeline is already explicit in code, versioned in git, and testable without a portal. A visual designer would add a second source of truth. Its *evaluation* SDK remains a good fit and is noted below. |
| **Azure AI Search Knowledge Agent** (agentic retrieval) | Not present in the installed SDK, and it duplicates query planning the pipeline needs anyway for conversational condensation. Worth revisiting when it is fully GA — it would replace `planner.py` and part of `searcher.py`. |
| **Azure Functions** | Container Apps fits better: the ingest workers are long-running and need warm clients, and cold starts are exactly wrong for a chat API. |
| **Azure Machine Learning** | Nothing here is trained. |
| **`azure-ai-evaluation`** | The harness needed metrics not in the SDK — abstention accuracy, citation resolution, per-category breakdown — and needed to run identically over two architectures. The SDK's `GroundednessEvaluator` and `DocumentRetrievalEvaluator` are a reasonable cross-check and are listed as future work rather than pretended into the results. |
| **Cosmos DB for Gremlin** | Managed and the right choice at scale; Cypher is materially clearer for the traversals here, and this build values that over the ops saving. Named in the production design as the migration target. |

---

## Cost shape

Measured on the eleven-document corpus with live services.

| Item | Measured | Notes |
|---|---|---|
| Graph extraction, whole corpus | **$0.034** | 2 LLM calls; 108 units, 38 triaged out |
| Extraction per 1000 documents | **$3.09** | A second run measured $3.27 — normal variance. Batch would halve it, once a batch-capable deployment exists |
| Re-ingest with no changes | **$0.00** | Content-hash memoization |
| Embeddings, whole corpus | negligible | 109 chunks at $0.02/1M tokens |
| Per query (improved pipeline) | ~$0.0025 | plan + answer + verify + judge |

The number that shapes the design: **output tokens are 96% of the extraction
bill.** Six of the seven cost controls are input-side, so the levers that
actually matter at scale are triage (do not extract from it), the memo cache
(do not extract from it twice), and Batch pricing. Prompt caching is real and
measured at 71%, and it saves about $0.003 of $0.034 — presenting it as the big
win would be misleading.
