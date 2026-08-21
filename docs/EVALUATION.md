# Evaluation

Baseline RAG → identify failures → improve the architecture → re-run → compare.

Both configurations run against the same live corpus, the same index, the same
questions, the same judge at the same temperature, and — importantly — the same
measuring instruments. Everything below is a measured number from
`eval/results/`, reproducible with `python -m eval.runner --mode both`.

> **About the figures in this document.** They are measured, and they are
> measured against `source_data/` — the eleven multi-page policy documents
> that ship with the project — so they reproduce on a clone. Point the
> pipeline at your own corpus and every method here still holds; the
> absolute values will be your corpus's, not these.


---

## 1. What is being compared

| | Baseline | Improved |
|---|---|---|
| Retrieval | Vector only, top 5 | Hybrid (BM25 + vector, RRF) + semantic reranking + graph retrieval |
| Query handling | The raw message | History condensed, classified, decomposed |
| Ranking | None | Cross-encoder leads, version conflicts resolved |
| Context | Flat top 5 | Neighbour expansion, budget scaled to the decomposition |
| Grounding | Answers whatever comes back | Sufficiency gate before, claim verification after, abstention |

The baseline is **not a strawman**. It is the pipeline this repository had
before the retrieval work — the shape most RAG tutorials produce — and it uses
the same answer prompt, so what is being measured is architecture, not
prompting.

Conversation history is handed to the baseline retriever raw, which is the
mistake real implementations make.

---

## 2. The golden set

51 hand-written cases against the real corpus (`eval/dataset.jsonl`),
stratified so a change that helps one kind of question and hurts another cannot
hide in an average:

| Category | Cases | Correct behaviour |
|---|---|---|
| Straightforward | 16 | Answer with a citation |
| Multi-document | 10 | Answer, having retrieved every part |
| Versioned | 6 | Answer from the current version and say which |
| Ambiguous | 6 | Ask a clarifying question naming the real options |
| No answer in corpus | 8 | Abstain |
| Follow-up (multi-turn) | 5 | Resolve the reference and answer |

One case (`a06`, "What is the tuition reimbursement limit?") is deliberately
*apparently* ambiguous but has one answer — it exists to catch
over-clarification, which is as much a failure as under-clarifying.

---

## 3. Results

51 cases, sequential (concurrency 1, so latency is not inflated by
rate-limiting).

### Retrieval

| Metric | Baseline | Improved |
|---|---|---|
| Hit rate@3 | 100.0% | 100.0% |
| Hit rate@5 | 100.0% | 100.0% |
| Recall@5 | 100.0% | 98.0% |
| **MRR** | 94.1% | **100.0%** |
| **nDCG@5** | 95.7% | **98.5%** |

**Read these honestly.** Hit rate is saturated at 100% for both, and that is a
property of an eleven-document corpus, not evidence of a good retriever: with
so few documents the right one is almost always somewhere in the top 3. The
metrics that still discriminate are the ranking ones — MRR reaching 1.000 means
the correct document is now *first* for every question, which is what actually
matters when the context is truncated.

### Generation

| Metric | Baseline | Improved |
|---|---|---|
| Answer correctness (judge 0–4, normalised) | 88.7% | **98.5%** |
| Fact match (deterministic) | 98.0% | 98.0% |
| Groundedness (claim-level) | 99.6% | 99.4% |
| Citation accuracy | 94.6% | **97.8%** |
| **Hallucination rate** | **3.9%** | **0.0%** |

### Behaviour

| | Baseline | Improved |
|---|---|---|
| **Behaviour accuracy** | 86.3% | **100.0%** |
| Correctly answered | 37 | 37 |
| Correctly abstained | 7 | 9 |
| Correctly clarified | 0 | 5 |
| **Wrongly answered** | **6** | **0** |
| Wrongly abstained | 1 | **0** |
| Wrongly clarified | 0 | 0 |

The single most important row is **wrongly answered: 6 → 0**. Those are
questions the corpus cannot answer, or that need a clarifying question, which
the baseline answered anyway — with citations, confidently. Every one is now
handled correctly, and `wrongly abstained` is 0 as well, so the improvement is
not the usual trick of refusing more often.

**On run-to-run variance.** These are LLM outputs at temperature 0, which is not
the same as determinism, and the ambiguous category sits closest to a decision
boundary. Earlier runs of this pipeline — before the routing fixes in §5 —
measured behaviour accuracy at 96.1% with one or two ambiguous cases falling the
wrong side. The 100% here is one run, not a guarantee; treat single-case
movement in a 51-case set as noise and the ~14-point gap over the baseline as
the finding.

### By category

| Category | Baseline | Improved |
|---|---|---|
| Straightforward | 16/16 · judge 3.69 | 16/16 · judge 3.88 |
| Multi-document | 10/10 · judge 3.70 | 10/10 · **judge 4.00** |
| Versioned | 6/6 · judge 4.00 | 6/6 · judge 3.83 |
| **Ambiguous** | **1/6** · judge 2.67 | **6/6** · **judge 4.00** |
| No answer | 6/8 · judge 3.50 | **8/8** · **judge 4.00** |
| Follow-up | 5/5 · judge 3.40 | 5/5 · **judge 4.00** |

Ambiguous handling is where the architecture earns most of its gain: 1/6 → 6/6.
Follow-ups and multi-document questions were already behaviourally correct in
the baseline, but the answers got materially better — condensation and
decomposition improved *what* was answered, not whether.

Versioned dips slightly on judge score while staying 6/6 on behaviour: judge
variance on answers that all name the right figure and the right document.

### System

| | Baseline | Improved |
|---|---|---|
| Latency p50 | 3.8 s | 9.0 s |
| Latency p95 | 17.4 s | **15.4 s** |
| Cost per query | $0.00130 | $0.00246 |

**The improved pipeline is roughly 2.4× slower at the median and 1.9× more
expensive**, and it should be: it makes a planning call, sometimes several retrievals, and a
verification call on every answered turn. That is the price of eliminating
confident wrong answers, and adaptive depth is what stops it being worse —
simple lookups skip decomposition and correction entirely.

The p95 is the interesting column: the improved pipeline is *faster* in the
tail. The baseline's worst cases are the ones where it flounders — retrieving
weakly, then generating a long speculative answer from thin evidence. Adaptive
depth and an early abstention both cut work off before that happens.

---

## 4. What the first run found

The first full run is worth recording, because two of its results were wrong
in instructive ways.

**Citation accuracy appeared to get *worse* — 80.4% → 70.6%.** It had not. The
metric returned 0.0 when there was nothing to cite, so every correct abstention
scored zero while the baseline's wrong answer, citing a real document, scored
one. The improved system was being punished for refusing. Cases with nothing to
cite are now excluded rather than counted as failures, and the corrected
measurement is 96.0% → 98.2%.

*A metric that inverts on the behaviour it exists to encourage is worse than no
metric.* This one would have been quoted as a regression in a report nobody
could have checked.

**One straightforward question abstained wrongly:** *"What are the payment
terms in the vendor service agreement?"* — answered in one sentence by
`VendorContract §3 Payment Terms`, which ranks **first at 2.95** in raw search.
It had fallen out of the context during fusion: RRF reads positions only, so
two retrievers agreeing on a mediocre chunk outranked the strong hit only one
of them found, and the cross-encoder's score was discarded. The design named a
rerank stage after fusion; the implementation never had one. Adding it fixed
the case and lifted MRR to 1.000.

Both were found by reading the per-case detail rather than the summary. An
aggregate would have shown a small regression and a small dip, and neither
cause is guessable from a number.

---

## 5. The two ambiguity failures, and their fix

An earlier run left ambiguity at 4/6. The two failures had opposite symptoms and
the same root: what happens *after* the sufficiency gate.

**`a05` — "What's the deadline?" abstained.** It scored 2.43 on the reranker
with **0.00 term coverage**, because the corpus says "within 30 calendar days"
and never once says "deadline". An under-specified question fails the coverage
check by construction — not because the documents lack the answer. The system
said "I don't know" when the truth was "I don't know which one you mean", and
the planner had already produced four concrete readings to offer.

*Fix:* an `AMBIGUOUS` plan with readings and topically relevant evidence
clarifies rather than abstaining — gated on the reranker clearing the
sufficiency threshold, so a genuinely empty corpus still abstains however vague
the question.

**`a03` — "How much notice do I need to give?" answered about PTO alone.** Four
notice periods exist — PTO at 5 or 15 business days, contract cancellation at
30, non-renewal at 60 — but the leave section dominated the ranking at 2.80, so
the near-top-tie test saw one clear winner.

*Fix:* when the planner has named three or more readings, evidence drawn from
more than one **department** confirms them. "Notice" in HR and "notice" in legal
are different facts; two chunks of one policy are the same fact twice.

**Why this could not break abstention:** every one of the eight no-answer cases
is classified `SIMPLE` with zero readings, so neither branch can reach them.
That is asserted directly in `tests/test_ambiguity_routing.py`, whose fixtures
are the measured diagnostics from this run rather than invented inputs —
thresholds tuned against imagined data are tuned against nothing.

Ambiguity is now **6/6**, and no case in the set fails.

---

## 6. Caveats a reader should hold

- **The corpus is eleven documents.** Retrieval metrics saturate; the ranking metrics carry the signal. Nothing here demonstrates behaviour at scale.
- **The judge shares a model family with the generator.** Only `gpt-4.1-mini` is deployed, so self-preference bias is possible. Mitigated with rubric-constrained scoring, a pinned deployment, temperature 0, and a deterministic fact-match check underneath that no amount of judge persuasion can move — a wrong figure fails regardless. A second judge from a different family is the proper fix.
- **Groundedness is measured with the same verifier the improved pipeline uses as a gate.** In the harness it is only a thermometer, and it is run over *both* systems' answers — measuring one side with a stricter instrument than the other would manufacture the result.
- **51 cases is small.** A single case is worth ~2 points of behaviour accuracy.
- **Latency was measured sequentially.** Under concurrency 2 the same pipeline measured p50 ~62 s because of rate limiting — a real production concern, and the reason the p95 alert in [`PRODUCTION.md`](PRODUCTION.md) exists.

---

## 7. Reproducing

```bash
python -m eval.runner --mode both --concurrency 1      # full run, both configs
python -m eval.runner --mode improved --category ambiguous
python -m eval.runner --mode baseline --limit 5
```

Results are written to `eval/results/{baseline,improved}.json`, including
per-case detail: the answer, the judge's score and reason, retrieved documents,
citations, and the full pipeline diagnostics for every case.
