# What I would change before production

This system runs against live Azure services and is honest about what it is: a
complete pipeline proven on eleven documents, designed so that scaling it is a
matter of swapping components rather than rewriting it. This document is the
gap list — what is deliberately shallow, what would break first, and the order
I would fix it in.

Nothing below is a surprise discovered late. Each item is a trade I made
knowingly, to get the whole system working end to end first: a capability that
does not exist cannot be improved, only rebuilt.

---

## The first five things, in order

### 1. Move the sidecars to Postgres

**Now:** the extraction cache, the document registry and LangGraph conversation
state are SQLite files on local disk.

**Problem:** correct for one process, wrong for two. The moment ingest runs on
more than one worker, they each keep their own cache and the memoization that
makes re-ingest free stops working across workers.

**Change:** one Postgres instance, same interfaces. `langgraph-checkpoint-postgres`
for conversation state. This is a day of work and it is the precondition for
everything else, which is why it is first.

### 2. Event-driven ingest with a dead-letter queue

**Now:** an inotify watcher and ETag polling inside the ingest process. A
document that fails is logged, counted and skipped.

**Problem:** polling does not scale and a skipped document is silently lost.
"We ingested 4,999,983 of 5,000,000 documents" is not an acceptable end state,
and right now nothing would tell you which seventeen.

**Change:** Blob → Event Grid → Service Bus → autoscaled workers, with failures
dead-lettered rather than dropped, a replay path, and an alert on failure rate.
`IngestError` already carries the doc id and the stage; it needs somewhere to go.

### 3. Real identity instead of a header

**Now:** the caller's departments arrive in `X-User-Departments`, which a client
could set for itself.

**Problem:** the enforcement is sound — deny by default, narrowing only,
filtered in the query — but the *input* is trusted. That is fine behind a
gateway and unacceptable without one.

**Change:** Entra ID authentication at API Management or Front Door, group
membership mapped to departments and injected server-side, with the header
stripped from inbound requests. For document-level control, index ADLS Gen2
ACLs and enforce them with `x-ms-query-source-authorization` so the source
system's permissions stay authoritative rather than being mirrored into a field
that drifts.

### 4. Secrets out of `.env`

**Now:** keys in a `.env` file, read by `rag.config`.

**Change:** Key Vault with managed identity, no keys in configuration at all.
`rag.config` is already the only thing that reads the environment, so this is
one class changing where it gets its values.

> `.env` is gitignored and must stay that way. If a key has ever been written
> into a tracked file, rotate it — `.gitignore` does not remove anything from
> history.

### 5. Backfill through the Batch API, and prove the cost model

**Now:** the batch path is implemented and its submission verified; online
extraction handles everything.

**Blocked on quota, no longer on provisioning.** A `datazonebatch` deployment
now exists, so `invalid_deployment_type` is gone. Submission now fails on
`token_limit_exceeded`: the deployment's enqueued-token limit is 1K and one
request is 2,297 tokens, most of it the deliberately-large cached prefix. The
corpus needs ~11,700. Raise the batch quota on the deployment and this runs.

**Change:** run the initial load in batch (50% off, 24-hour window, separate
quota), keep online for the change stream. Before committing to a budget,
benchmark on a real sample of the target corpus — 5M documents of 100–200
pages is a very different token profile from eleven two-page policies, and the
$3.09-per-1000 figure measured here should be treated as a method that
reproduces, not a number that transfers.

---

## What breaks at scale, and when

| Scale | What breaks | Fix |
|---|---|---|
| ~50k docs | Single-process ingest is too slow | Queue-driven workers |
| ~100k docs | Vector quota on one partition | Add partitions; scalar quantization (~4× smaller) |
| ~500k docs | Entity resolution over the whole set per ingest | Batch resolution offline, incremental blocking |
| ~1M docs | Neo4j single instance write throughput | Cluster, or Cosmos DB Gremlin |
| ~1M docs | In-process query cache per replica | Redis, still scope-partitioned in the key |
| ~5M docs | Backfill cost and duration | Batch API, binary quantization (up to ~28× smaller), staged rollout by department |
| Any scale | A pipeline change forces a full re-index | Version the pipeline, stamp the version on every chunk, run two versions side by side |

That last row is the one that gets underestimated. At five million documents
you *will* change the chunker, the embedding model or the ontology at least
once, and a full re-index is weeks of compute and a large bill. The
content-hash cache and per-document memoization here are the beginning of the
answer; a production system needs the pipeline version in the cache key so a
change invalidates exactly what it affects and nothing else.

---

## Observability I would add on day one

The counters at `GET /stats` and the per-turn `diagnostics` are enough to debug
a single request. Production needs the aggregate view, and specifically these
four alerts — chosen because each one catches a failure that is otherwise
silent:

| Alert | Catches |
|---|---|
| **p95 latency**, not mean | Rate limiting. A 429 with a transparent SDK retry looks like the model got slower and errors nowhere. Measured here: p50 went from ~12s to ~62s at concurrency 2 with no code change. |
| **Cost per query**, not total spend | A regression. Total spend rises with traffic; cost per query is the number that reveals a `top_k` change or an extra model call that nobody noticed. |
| **Abstention rate** | Both directions. A rise means retrieval broke or the corpus lost a document. A fall toward zero means the grounding gate stopped working — and that failure looks like an improvement on every other metric. |
| **Hallucination rate** from live verification | The silent failure. Verification already runs on every answered turn; emitting its verdict as a metric turns "occasionally a wrong answer" from an anecdote into a number. |

Full traces to Application Insights, with the retrieved chunk ids on the span.
Without them you cannot replay a turn, and without a replay you are guessing.

---

## Known limitations, stated plainly

- **Proven on eleven documents.** Every number in these docs is measured, and every number is measured on a small clean corpus. The architecture is designed for 5M; the *evidence* covers eleven. Claims about 5M behaviour are labelled as projections throughout.
- **The batch path is one quota slider away from running.** A `datazonebatch` deployment now exists and the SKU rejection is gone; submission fails on `token_limit_exceeded` instead — the deployment's **enqueued-token quota is 1K**, and the smallest possible request is **2,297 tokens** because the cache-friendly system prefix is 2,263 of them. The whole corpus needs **~11,700 enqueued tokens across 3 packs**. Raising the deployment's batch quota (a portal setting, separate from the online TPM allocation) is the only remaining step; until then `apply()` is covered by recorded payloads rather than a live round trip, and the 50% saving stays a projection.
- **The evaluation judge is the same model family as the generator.** Only `gpt-4.1-mini` is deployed, which risks self-preference bias. Mitigated with rubric-constrained scoring, a pinned deployment, temperature 0, and a deterministic fact-match check underneath the judge that no amount of persuasion can move — a wrong figure fails regardless of what the judge says. A second judge from a different family is the proper fix.
- **Semantic caching is in-process.** Correct for one replica; it needs Redis and the same scope-partitioned key before a second one exists.
- **No PII detection or redaction.** The corpus is internal policy documents. A corpus containing personal data needs Presidio or Content Safety in the ingest path, not just at query time.
- **Figure captioning is off by default.** Charts and diagrams are OCR'd for text but not described. `VISION_CAPTIONS_ENABLED` turns it on at one vision call per image.
- **Single region.** No failover, no geo-replication.
- **No load test.** Concurrency behaviour is inferred from the rate-limiting observed during evaluation runs, not measured under sustained load.

---

## What I would not change

Worth stating, because "productionise it" often means rewriting things that are
already right:

- **Deny-by-default scoping enforced in the query.** The mechanism is correct; only its input needs to become trustworthy.
- **Two independent grounding gates.** Sufficiency before generation, verification after. They fail independently, which is the point.
- **Provenance on every extracted relation.** This is what makes an LLM-built graph auditable instead of merely plausible, and it costs nothing at query time.
- **Superseded documents stay indexed.** Deleting them would make legitimate historical questions unanswerable.
- **Triage before extraction.** The single largest cost control, and it gets more valuable as the corpus grows, not less.
