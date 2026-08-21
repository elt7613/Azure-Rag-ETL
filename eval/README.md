# Evaluation harness

Measures the retrieval pipeline against a golden set, and compares two
architectures under identical conditions so a change can be shown to help
rather than assumed to.

```bash
python -m eval.runner --mode both --concurrency 1   # baseline vs improved
python -m eval.runner --mode improved               # just the full pipeline
python -m eval.runner --mode improved --category ambiguous
python -m eval.runner --mode baseline --limit 5     # quick smoke
```

Results land in `eval/results/{baseline,improved}.json`, including per-case
detail: the answer, the judge's score and reasoning, retrieved documents,
citations, and the full pipeline diagnostics.

`--concurrency 1` matters when you care about the latency numbers — running
cases in parallel triggers rate limiting, which inflates them without anything
in the code having changed.

---

## The dataset is an example — replace it

`dataset.jsonl` is written against the corpus in `source_data/`. It
demonstrates the format and the category mix; **it will not measure anything
useful about your own documents.** Replace it with questions about your corpus.

One JSON object per line:

```json
{
  "id": "s01",
  "category": "straightforward",
  "difficulty": "easy",
  "expected_behavior": "answer",
  "question": "How many days of paid sick leave do employees get per year?",
  "expected_answer": "12 days per calendar year, credited in full on 1 January.",
  "expected_docs": ["HR/LeavePolicy.pdf"],
  "expected_sections": ["3 Sick Leave"],
  "must_contain": ["12"],
  "departments": ["HR"]
}
```

| Field | Purpose |
|---|---|
| `id` | Stable identifier; used in reports so a regression can be traced to a case |
| `category` | Groups the per-category breakdown. Use your own if the defaults do not fit |
| `expected_behavior` | `answer`, `abstain` or `clarify` — scored as a confusion matrix |
| `expected_answer` | Reference text the LLM judge scores the response against |
| `expected_docs` | Document ids retrieval should return. Drives hit rate, recall, MRR and nDCG |
| `must_contain` | Literal strings the answer must include. A deterministic check underneath the judge — a wrong figure fails regardless of how confident the answer sounds |
| `match_any` | Optional. When true, any one `must_contain` term satisfies it (for facts the documents state in more than one form) |
| `departments` | Access scope for the request. Omit to use every configured department |
| `history` | Optional list of `{role, content}` turns, for testing follow-up questions |

### Writing a set that measures something

- **Stratify it.** A single average hides a change that helps one kind of question and hurts another. The example set uses straightforward, multi-document, versioned, ambiguous, no-answer and follow-up.
- **Include questions your corpus cannot answer.** Without them you are not measuring hallucination at all, only accuracy on questions that happen to have answers.
- **Include at least one question that *looks* ambiguous but is not.** Over-clarifying is as much a failure as under-clarifying, and nothing else in the set catches it.
- **Keep `must_contain` to figures and names**, not phrasing. It exists to be unfakeable, and a judge can be talked into accepting a wrong number where a string match cannot.

---

## What is measured

**Retrieval** — hit rate@{3,5,10}, recall@5, MRR, nDCG@5. On a small corpus hit
rate saturates near 100% and stops discriminating; the ranking metrics are the
ones carrying signal.

**Generation** — answer correctness (rubric-constrained LLM judge), fact match
(deterministic), groundedness (claim decomposition against the cited passages),
citation accuracy, hallucination rate.

**Behaviour** — a confusion matrix over answer/abstain/clarify. Answering an
unanswerable question and refusing an answerable one are different failures
with different remedies, and folding them into one accuracy number rewards
over-refusal.

**System** — p50/p95 latency, prompt and completion tokens, estimated cost per
query.

---

## The two configurations

| | `baseline` | `improved` |
|---|---|---|
| Retrieval | Vector only, top 5 | Hybrid + semantic reranking + graph |
| Query handling | The raw message | Condensed, classified, decomposed |
| Ranking | None | Cross-encoder leads; version conflicts resolved |
| Grounding | Answers whatever comes back | Sufficiency gate, verification, abstention |

Both use the same corpus, the same judge at the same temperature, and the same
measuring instruments — groundedness is measured by running the verifier over
*both* systems' answers, because using a stricter instrument on one side would
manufacture the result.

A caveat worth keeping in mind: the judge shares a model family with the
generator, which risks self-preference bias. It is mitigated with
rubric-constrained scoring, a pinned deployment, temperature 0, and the
deterministic `must_contain` check underneath. A judge from a different family
is the proper fix.
