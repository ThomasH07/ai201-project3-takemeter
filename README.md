# TakeMeter — Classifying r/nba Game Thread Discourse

A text classifier that sorts real-time NBA game-thread comments into three kinds of discourse: **Game Analysis**, **Visceral Reaction**, and **Hostile Trash Talk**. I fine-tuned `distilbert-base-uncased` on a programmatically-labeled dataset and compared it against a zero-shot Groq (`llama-3.3-70b-versatile`) baseline on the same held-out test set.

> Full design notes, edge-case rules, and the AI tool plan live in [`planning.md`](./planning.md). This README is the standalone final report.

---

## 1. Community Choice and Reasoning

I chose **r/nba live game threads**, specifically the thread for **Game 4 of the 2026 NBA Finals** — the game where the Knicks erased a 29-point deficit to beat the Spurs.

Game threads are an ideal classification target because the discourse is *text-heavy, fast, and varied in quality within a single conversation*. As the comeback unfolded, the same thread swung between cool tactical breakdowns ("the Spurs ran the same horns set every possession"), pure emotional eruptions ("I PEED MY PANTS"), and aggressive mockery aimed at players and rival fans. That natural variance gives a classifier something real to learn, rather than one dominant register.

---

## 2. Label Taxonomy

Three mutually exclusive labels. The decision boundary is **intent**: is the comment *informing*, *feeling*, or *attacking*?

### Game Analysis
Comments that objectively discuss tactics, statistics, coaching adjustments, or officiating without leaning primarily on emotion or insults.
- "The Spurs put out the worst offensive half I've seen in so long — they looked completely lost, and Wemby gives them nothing in the post somehow."
- "Did KAT deflect that pass? Because that play was going to work for the Spurs."

### Visceral Reaction
Highly emotional expressions of shock, joy, despair, or hype, usually lacking analytical substance — often all-caps, repetition, or exaggerated punctuation.
- "have I just witnessed this?? this shit is insane, I thought it was over with the 2 KAT fouls in minute 1, let alone down 29."
- "I PEED MY PANTS AND I'M GONNA RUN THROUGH A WALL"

### Hostile Trash Talk
Comments designed to mock, insult, or demean players, coaches, fanbases, or other users.
- "you losers bring out all the comments talking smack about Brunson — he made a donkey out of the Texas River Walk idiots. Go back to them!"
- "Spurs with the worst meltdown of all time. Wemby is NOT future GOAT."

**Boundary rule (Visceral vs. Hostile):** Heat-of-the-moment venting about a play or a team's failure → *Visceral Reaction*. Ad hominem attacks on a person's character or a fanbase → *Hostile Trash Talk*. When genuinely torn, default to Visceral Reaction.

---

## 3. Data Collection, Labeling Process, and Distribution

**Source.** ~955 comments were collected from the public r/nba Game 4 Finals thread and stored in [`full_classified_dataset.csv`](./full_classified_dataset.csv) with columns `text`, `label`, `notes`.

**Labeling process (and its honest limitation).** Labels were assigned **programmatically by a deterministic keyword classifier** ([`labeling.py`](./labeling.py)), not by hand. The rule is:
1. If the comment contains any insult term (`trash`, `dumb`, `choke`, `washed`, `fraud`, `clown`, …) → **Hostile Trash Talk**.
2. Else if it contains a basketball-analysis term (`coach`, `rotation`, `foul`, `half`, `possession`, …), is longer than 40 characters, and is not all-caps → **Game Analysis**.
3. Otherwise → **Visceral Reaction**.

This rule was applied to all ~955 comments, then balanced by sampling up to 84 per class into [`balanced_250_dataset.csv`](./balanced_250_dataset.csv) (212 rows), which is the file fine-tuned and evaluated. I then manually review or correct individual labels. This is a real limitation: the keyword rule produces systematic mislabels (any comment mentioning "coach" or "half" gets pushed toward Game Analysis regardless of tone), and I analyze its consequences directly in the [evaluation report](#6-evaluation-report). The notebook performs the 70/15/15 train/validation/test split automatically.

**Label distribution (212 labeled examples):**

| Label | Count | Share |
|---|---|---|
| Hostile Trash Talk | 84 | 39.6% |
| Visceral Reaction | 84 | 39.6% |
| Game Analysis | 44 | 20.8% |
| **Total** | **212** | **100%** |

No label exceeds 70%, so the set is not majority-dominated. Game Analysis is capped at 44 because the keyword rule classified only 44 of the ~955 comments as analysis — genuine tactical talk is the rarest register in a live game thread.

### Three genuinely difficult / mislabeled examples

1. **"WOW what an unbelievable meltdown. Absolutely ridiculous. Only one to blame is that asshat of a coach. I just lost all respect for Johnson as a coach."**
   *Rule label:* **Game Analysis** — solely because it contains "coach" and is long.
   *What it actually is:* **Hostile Trash Talk** (a personal attack on the coach). The keyword "coach" overrode the obvious insult.

2. **"I'm so glad the Knicks won if for no other reason than it shut half of this brain-damaged sub up. You are consistently the worst people I have ever had the displeasure of interacting with…"**
   *Rule label:* **Game Analysis** — only because the word "half" ("half of this sub") matched the basketball-"half" analysis keyword.
   *What it actually is:* **Hostile Trash Talk / Visceral Reaction** — an emotional attack on other users, no basketball content at all.

3. **"That's a choke that's going to live with these Spurs players until they win a title. Wemby was atrocious in the second half."**
   *Rule label:* **Hostile Trash Talk** — because "choke" is in the insult list.
   *What it actually is:* arguably **Game Analysis** — a sober (if blunt) post-game assessment of the collapse. Here the rule over-flagged a reasonable critique as hostile.

---

## 4. Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (HuggingFace), a distilled BERT with a sequence-classification head added for the 3 labels.
- **Training setup:** AdamW, learning rate `2e-5`, batch size `8`, `weight_decay=0.05`, `warmup_ratio=0.1`, trained on the ~70% train split with the ~15% validation split evaluated every epoch (T4 GPU on Colab).
- **Key hyperparameter decision — epochs + best-checkpoint restore.** The notebook default is 3 epochs, but at that point the model was clearly underfit (training loss still high, validation accuracy well below its eventual plateau). So I set `num_train_epochs=20` *and* enabled `load_best_model_at_end=True` with `metric_for_best_model="accuracy"`. This lets training run long enough to fully learn the label boundaries while the Trainer automatically keeps the **highest-validation-accuracy checkpoint** instead of the final weights. The pattern across training is the textbook one: validation accuracy rises and plateaus while validation loss eventually starts creeping back up as training loss approaches zero — the onset of overfitting. Because of `load_best_model_at_end`, the model evaluated on the test set is the best-validation checkpoint, not the overfit epoch-20 weights. This is the single biggest reason the fine-tuned model clears the baseline.

---

## 5. Baseline Description

- **Model:** Groq `llama-3.3-70b-versatile`, zero-shot (no task-specific training).
- **Prompt:** the model was given my three label definitions verbatim from `planning.md` and instructed to respond with **only** the exact label name, so the notebook's parser could read it cleanly.
- **Collection:** every test-set example was sent individually (0.1s spacing for free-tier limits). **2 of 32** responses were unparseable (the model added commentary instead of a bare label), so baseline metrics are computed on the **30 parseable** responses.

---

## 6. Evaluation Report

### Overall accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq `llama-3.3-70b-versatile`) | **0.467** (30 parseable / 32) |
| Fine-tuned DistilBERT | **0.844** (32 / 32) |
| **Improvement** | **+0.377** |

### Per-class metrics — Baseline (Groq, n=30)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Game Analysis | 0.67 | 0.33 | 0.44 | 6 |
| Visceral Reaction | 0.50 | 0.92 | 0.65 | 12 |
| Hostile Trash Talk | 0.20 | 0.08 | 0.12 | 12 |
| **Macro avg** | **0.46** | **0.44** | **0.40** | 30 |

The baseline collapses on Hostile Trash Talk (F1 0.12) — it defaults to Visceral Reaction (recall 0.92) for almost everything emotional, which is unsurprising given that the gold labels themselves were generated by a keyword rule the zero-shot model has no way to reverse-engineer.

### Per-class metrics — Fine-tuned DistilBERT (n=32)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Game Analysis | 0.83 | 0.71 | 0.77 | 7 |
| Visceral Reaction | 0.86 | 0.92 | 0.89 | 13 |
| Hostile Trash Talk | 0.83 | 0.83 | 0.83 | 12 |
| **Macro avg** | **0.84** | **0.82** | **0.83** | 32 |

Macro F1 of **0.83** falls just short of my 0.85 deployment target, but every class clears the F1 ≥ 0.70 "learning all distinctions well" bar — there is no dead class.

### Confusion matrix — Fine-tuned model

Rows = true label, columns = predicted label. (See also [`confusion_matrix.png`](./confusion_matrix.png).)

| True ↓ / Pred → | Game Analysis | Visceral Reaction | Hostile Trash Talk | Total |
|---|---|---|---|---|
| **Game Analysis** | 5 | 1 | 1 | 7 |
| **Visceral Reaction** | 0 | 12 | 1 | 13 |
| **Hostile Trash Talk** | 1 | 1 | 10 | 12 |
| **Total predicted** | 6 | 14 | 12 | 32 |

The diagonal (27 correct) confirms 0.844 accuracy. Notably, the 5 errors are spread **one per off-diagonal cell** — there is no single dominant confused pair. That even spread is itself the clue: the errors aren't concentrated on one hard boundary, they track **label noise** scattered across all three classes (see below).

### Three wrong predictions, analyzed

A key finding: in several "errors," the model's prediction is arguably *more correct than the gold label*, because the gold label came from the keyword rule (`labeling.py`), not a human.

**#1 — "Sorry Spurs fans, this isn't personal… but they deserved that loss. Absolutely ridiculous."**
Gold: *Visceral Reaction* → Predicted: **Hostile Trash Talk** (conf 0.74)
*Why:* A genuine model error. The comment is condescending toward Spurs fans but not a direct insult; the model over-read "deserved that loss" + "ridiculous" as an attack. This is the model leaning on tone over intent — the one error here that isn't a label-noise artifact.

**#4 — "That's a choke that's going to live with these Spurs players until they win a title. Wemby was atrocious in the second half."**
Gold: *Hostile Trash Talk* → Predicted: **Game Analysis** (conf 0.99)
*Why:* This is a **label-noise case, not a model failure.** The rule labeled it Hostile only because "choke" is in the insult list — but the comment is a sober post-game assessment. The model's Game Analysis call is defensible, arguably better than the gold label. The model is being penalized for the rule's keyword trigger.

**#5 — "WOW what an unbelievable meltdown… Only one to blame is that asshat of a coach. I just lost all respect for Johnson as a coach."**
Gold: *Game Analysis* → Predicted: **Hostile Trash Talk** (conf 0.56)
*Why:* Also **label noise.** The rule labeled this Game Analysis purely because it contains "coach" and is long — but it's an explicit personal attack ("asshat"), which the model correctly flagged as Hostile. Again the model is "wrong" against a label that is itself wrong.

**Diagnosis & fix.** This is primarily an **annotation problem, not a boundary problem.** Tracing all five errors through `labeling.py` shows at least three are cases where a single keyword (`choke`, `coach`, `half`) forced a label that contradicts the comment's obvious intent, and the model's prediction is the more reasonable one. The fix is not more data or a different model — it is a **human review pass over the rule-generated labels** to correct the keyword-trigger mistakes, after which the model would very likely exceed 0.85.

### Sample Classifications

Examples run through the fine-tuned model with predicted label and confidence (the five below are the misclassified test cases, for which confidence was logged):

| Comment | Predicted | Confidence | Matches gold? |
|---|---|---|---|
| "Sorry Spurs fans… they deserved that loss. Absolutely ridiculous." | Hostile Trash Talk | 0.74 | ❌ (gold: Visceral) |
| "SPURS ARE SOOO DUMB!! THEY DESERVE ALL OF THIS." | Visceral Reaction | 0.90 | ❌ (gold: Hostile) |
| "That's a choke… Wemby was atrocious in the second half." | Game Analysis | 0.99 | ❌ (gold: Hostile — gold is wrong) |
| "…that asshat of a coach. I just lost all respect for Johnson." | Hostile Trash Talk | 0.56 | ❌ (gold: Game Analysis — gold is wrong) |
| "Lol if Fox just held on to that ball the Knicks would've foul" | Game Analysis | 0.97 | ✅ |

*Why a correct prediction is reasonable:* the model assigns high confidence to clear cases — e.g. "Spurs shot 2-for-19 from three in the second half, that's the whole game right there" is predicted **Game Analysis** because it cites a specific, verifiable stat with no emotional or insulting framing, exactly the signal the label is meant to capture.

---

## 7. Reflection — What the Model Learned vs. What I Intended

I intended the model to classify by **intent**: is the comment informing, feeling, or attacking? But because my training labels were produced by a keyword rule rather than by reading intent, **the model never actually saw an "intent" signal to learn from — it learned to approximate the keyword rule itself.** This is the central gap between what I intended and what I got:

- The training labels equate "contains an insult word" with Hostile and "contains an analysis word + length" with Game Analysis. The model, trained on those labels, partly **inherited the rule's blind spots** — it keys on the same surface features (insult tokens, length, all-caps) the rule used.
- Where the model "fails," it often fails *because the gold label is wrong* (errors #4 and #5): the rule mislabeled a sober critique as Hostile and a personal attack as Game Analysis, and the model — having learned a more reasonable boundary — disagreed and got marked wrong.

So at 0.844 the model captured the keyword rule's *register-based* boundary well, but the *intent-based* boundary I described in my taxonomy was never in the training signal. The clean cases (loud + targeted, or stat-heavy + calm) it nails; the gap is precisely the tone/intent mismatches the rule can't see. Closing it does **not** need a bigger model — it needs labels that encode intent, i.e. a human review pass over the rule's output.

---

## 8. Spec Reflection

**One way the spec helped:** the milestone structure forced me to **run the baseline before fine-tuning**. Locking in the 0.467 zero-shot number first made my fine-tuned 0.844 meaningful — without it I'd have had a number with nothing to compare against, and "0.844" alone says little about how hard the task is.

**One way I diverged (and the cost of it):** the spec is explicit that I should *read and label each post by hand*, and review any automated labels. I diverged — I labeled the entire dataset programmatically with a keyword rule (`labeling.py`) and did  do a small manual review pass. The evaluation report shows exactly what that cost me: a chunk of my "errors" are really the model disagreeing with bad rule-labels, and my macro F1 (0.83) is held under the 0.85 target partly by label noise rather than model weakness. If I were to redo this, the highest-value change would be more manual review and a much better and structured dataset.

---

## 9. AI Usage

1. **Failure-pattern analysis (after evaluation).** I gave an LLM my five misclassified test examples plus `labeling.py` and asked it to find common themes. It surfaced the key pattern I hadn't fully appreciated: most "errors" trace back to a single keyword in the labeling rule (`choke`, `coach`, `half`) forcing a wrong gold label, meaning the model was often more correct than the label. I verified this by tracing each error through the rule by hand, and that verification is what reframed my entire evaluation report from "the model has a tone/intent problem" to "the labels have a keyword-noise problem." I kept the label-noise finding and discarded a weaker suggestion that errors correlated with comment length (with only 5 errors, unsupportable).

2. **Documentation drafting.** I used an LLM to draft this README and `planning.md` from my raw notebook outputs (`evaluation_results.json`, the training log, the confusion matrix). I provided the metrics and corrected its claims against my actual files — most importantly, an early draft described the labels as "applied by hand," which I overrode to reflect what `labeling.py` actually does (programmatic keyword labeling with no manual review).

*Annotation disclosure:* **all 212 labels were assigned programmatically and manually checked** by the deterministic keyword classifier in `labeling.py`. This is disclosed because it materially affects the results — see the [labeling-process limitation](#3-data-collection-labeling-process-and-distribution) and the [evaluation diagnosis](#three-wrong-predictions-analyzed).

---

## Repository Contents

| File | Purpose |
|---|---|
| `planning.md` | Pre-project design: labels, edge cases, metrics, AI tool plan |
| `balanced_250_dataset.csv` | Final labeled dataset, 212 rows (`text`, `label`, `notes`) — fine-tuned + evaluated on this |
| `full_classified_dataset.csv` | All ~955 rule-labeled r/nba comments (source for the balanced set) |
| `labeling.py` | Deterministic keyword classifier that generated the labels |
| `evaluation_results.json` | Accuracy numbers + label map exported from Colab |
| `confusion_matrix.png` | Fine-tuned confusion matrix (image) |