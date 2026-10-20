## Sentiment track (bake-off candidate #4)

Scored separately from the CONFIRM/VETO leaderboard: this is a
three-class labelling task, not a veto decision, so the two are not
comparable on one scale.

- Dataset: 60 hand-written headlines (19 marked *hard*: negation, mixed signals, sarcasm, jargon)
- Label balance: {'NEGATIVE': 24, 'POSITIVE': 18, 'NEUTRAL': 18}
- Always-predict-majority baseline: **40.0%** accuracy

> **The dataset is hand-written, not scraped.** The news collector is
> roadmap Phase 6 and does not exist yet, so there is no real feed to
> sample. These scores are a sanity check on the classifier, not an
> estimate of production accuracy. See `evaluation/sentiment_dataset.py`.

| Classifier | Kind | Available | Accuracy | Hard-subset accuracy | Macro F1 | Mean latency |
|---|---|---|---|---|---|---|
| `finbert/finbert` | local | yes | 63.3% | 47.4% | 0.604 | 50.7ms |
| `ollama/qwen3:1.7b` | local | no - Ollama not reachable at http://localhost:11434 (<urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>) | n/a | n/a | 0.000 | n/a |
| `lexicon-baseline` | baseline | yes | 63.3% | 26.3% | 0.634 | 0.3ms |

### Reading this

- Best available: **`finbert/finbert`** at 63.3% overall, 47.4% on the hard subset.
- Compare against the 40.0% majority-class baseline: anything near that number has learned nothing.
- The hard subset is the one that matters. Easy headlines are easy for
  a word list too; the question is whether a learned model handles
  negation and mixed signals better than keyword counting.

### Per-class detail

**`finbert/finbert`**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| POSITIVE | 0.615 | 0.444 | 0.516 | 18 |
| NEGATIVE | 0.778 | 0.875 | 0.824 | 24 |
| NEUTRAL | 0.450 | 0.500 | 0.474 | 18 |

**`lexicon-baseline`**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| POSITIVE | 0.625 | 0.556 | 0.588 | 18 |
| NEGATIVE | 0.941 | 0.667 | 0.780 | 24 |
| NEUTRAL | 0.444 | 0.667 | 0.533 | 18 |
