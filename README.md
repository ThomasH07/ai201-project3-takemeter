# TakeMeter — Classifying r/nba Game Thread Discourse

A text classifier that sorts real-time NBA game-thread comments into three kinds of discourse: **Game Analysis**, **Visceral Reaction**, and **Hostile Trash Talk**. I fine-tuned `distilbert-base-uncased` on hand-labeled comments and compared it against a zero-shot Groq (`llama-3.3-70b-versatile`) baseline on the same held-out test set.

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

**Source.** Comments were scraped from the public r/nba Game 4 Finals thread and staged in [`dataset.csv`](./dataset.csv) (~955 raw comments carrying Reddit's own `sentiment_label` of negative/neutral/positive). The raw sentiment field was used only as a *sampling aid* to pull a mix across emotional registers — it is **not** my label.

**Labeling process.** I re-read each sampled comment and applied the three-class taxonomy by hand using the definitions and boundary rule above, recording difficult calls as I went. The final labeled file is [`balanced_200_dataset.csv`](./balanced_200_dataset.csv) with columns `text`, `label`, `notes`. The notebook performs the 70/15/15 train/validation/test split automatically.

**Label distribution (178 labeled examples):**

| Label | Count | Share |
|---|---|---|
| Hostile Trash Talk | 67 | 37.6% |
| Visceral Reaction | 65 | 36.5% |
| Game Analysis | 46 | 25.8% |
| **Total** | **178** | **100%** |

No label exceeds 70%, so the set is not majority-dominated.

### Three genuinely difficult examples

1. **"Brunson is literally throwing this game away into the trash can, I can't watch this garbage anymore."**
   *Conflict:* "trash"/"garbage" reads like Hostile Trash Talk, but it's a fan despairing over a play.
   *Decision:* **Visceral Reaction** — heat-of-the-moment venting, not a malicious character attack.

2. **"Thibs needs to be investigated by the FBI for terrorism with these rotations."**
   *Conflict:* mentions an analytical concept ("rotations") but wraps it in hyperbolic insult.
   *Decision:* **Visceral Reaction** — the point is exaggeration/shock, not genuine analysis or targeted malice.

3. **"Popovich is completely washed, hasn't known how to coach since Duncan left, and the Spurs are a poverty franchise."**
   *Conflict:* coaching talk leans Game Analysis, but the tone is openly demeaning.
   *Decision:* **Hostile Trash Talk** — "washed" + "poverty franchise" crosses from venting into intentional malice.

---

## 4. Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (HuggingFace), a distilled BERT with a sequence-classification head added for the 3 labels.
- **Training setup:** AdamW, learning rate `2e-5`, batch size `8`, `weight_decay=0.05`, `warmup_ratio=0.1`, trained on the ~70% train split with the ~15% validation split evaluated every epoch (T4 GPU on Colab).
- **Key hyperparameter decision — epochs + best-checkpoint restore.** The notebook default is 3 epochs, but at epoch 3 validation accuracy was only **0.59** with training loss still high — clearly underfit. So I set `num_train_epochs=20` *and* enabled `load_best_model_at_end=True` with `metric_for_best_model="accuracy"`. This lets training run long enough to fully learn the label boundaries while the Trainer automatically keeps the **highest-validation-accuracy checkpoint** instead of the final weights. The validation curve confirms why this matters: accuracy climbed to **0.85 at epoch 10** and then plateaued, while validation loss began rising after ~epoch 7 as training loss approached zero — the onset of overfitting. The restored checkpoint (epoch 10, val acc 0.852) is therefore the model I evaluate on the test set, not the overfit epoch-20 weights. This is the single biggest reason the fine-tuned model clears the baseline.

---

## 5. Baseline Description

- **Model:** Groq `llama-3.3-70b-versatile`, zero-shot (no task-specific training).
- **Prompt:** the model was given my three label definitions verbatim from `planning.md` and instructed to respond with **only** the exact label name, so the notebook's parser could read it cleanly.
- **Collection:** every test-set example was sent individually (0.1s spacing for free-tier limits). **3 of 27** responses were unparseable (the model added commentary instead of a bare label), so baseline metrics are computed on the **24 parseable** responses.

---

## 6. Evaluation Report

### Overall accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq `llama-3.3-70b`) | **0.542** (24 parseable / 27) |
| Fine-tuned DistilBERT | **0.815** (27 / 27) |
| **Improvement** | **+0.273** |

### Per-class metrics — Baseline (Groq, n=24)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Game Analysis | 1.00 | 0.60 | 0.75 | 5 |
| Visceral Reaction | 0.47 | 0.78 | 0.58 | 9 |
| Hostile Trash Talk | 0.50 | 0.30 | 0.38 | 10 |
| **Macro avg** | **0.66** | **0.56** | **0.57** | 24 |

### Per-class metrics — Fine-tuned DistilBERT (n=27)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Game Analysis | 0.86 | 0.86 | 0.86 | 7 |
| Visceral Reaction | 0.78 | 0.70 | 0.74 | 10 |
| Hostile Trash Talk | 0.82 | 0.90 | 0.86 | 10 |
| **Macro avg** | **0.82** | **0.82** | **0.82** | 27 |

Macro F1 of **0.82** falls just short of my 0.85 deployment target, but every class clears the F1 ≥ 0.70 "learning all distinctions well" bar — there is no dead class.

### Confusion matrix — Fine-tuned model

Rows = true label, columns = predicted label. (See also [`confusion_matrix.png`](./confusion_matrix.png).)

| True ↓ / Pred → | Game Analysis | Visceral Reaction | Hostile Trash Talk | Total |
|---|---|---|---|---|
| **Game Analysis** | 6 | 1 | 0 | 7 |
| **Visceral Reaction** | 1 | 7 | 2 | 10 |
| **Hostile Trash Talk** | 0 | 1 | 9 | 10 |
| **Total predicted** | 7 | 9 | 11 | 27 |

The diagonal (22 correct) confirms 0.815 accuracy. Off-diagonal errors cluster on the **Visceral ↔ Hostile** boundary (3 of 5 errors), exactly the boundary I flagged as hardest in `planning.md`.

### Three wrong predictions, analyzed

**#1 — "WHAT THE HELL JUST HAPPENED!!!!???? OMGGGGGG"**
True: *Visceral Reaction* → Predicted: **Hostile Trash Talk** (conf 0.62)
*Why:* "WHAT THE HELL" + all-caps + heavy punctuation are surface features the model has learned to associate with hostility. It latched onto the aggressive *form* and missed that there's no target — pure shock, not an attack. This is the model over-weighting tone over intent.

**#3 — "Spurs playing historically stupid basketball in the 2nd half"**
True: *Game Analysis* → Predicted: **Visceral Reaction** (conf 0.66)
*Why:* This is a real (if blunt) tactical observation about second-half play, but the word "stupid" and the hyperbolic "historically" look like an emotional outburst. The structure is analytical; the vocabulary is emotional, and the model followed the vocabulary. A labeling-vs-data question: this is borderline even for a human, and I have few analytical comments that use casual/insulting vocabulary, so the model never learned that analysis can sound harsh.

**#5 — "Suck a fat one, Wemby!"**
True: *Hostile Trash Talk* → Predicted: **Visceral Reaction** (conf 0.70)
*Why:* Short, exclamatory, and slangy — it reads like a hype outburst, and the model leans on length/punctuation cues. It missed the second-person insult directed at a named player, which is the defining feature of Hostile Trash Talk. Short hostile one-liners are underrepresented in training, so the boundary is under-learned in that direction.

**Diagnosis & fix.** The errors are a **boundary problem, not an annotation problem** — I labeled these consistently with my rules; the model simply hasn't fully learned that *intent* outranks *tone*. The fix is more training examples of the two awkward cases: (a) analytical comments that use harsh/casual vocabulary, and (b) short, targeted insults — so the model stops using punctuation and all-caps as a proxy for the label.

### Sample Classifications

Examples run through the fine-tuned model with predicted label and confidence:

| Comment | Predicted | Confidence | Correct? |
|---|---|---|---|
| "This comeback is like, top 3 ball moment of all time for me lmao" | Game Analysis | 0.85 | ❌ (true: Visceral) |
| "Daaaamn absolute insane comeback" | Hostile Trash Talk | 0.79 | ❌ (true: Visceral) |
| "Suck a fat one, Wemby!" | Visceral Reaction | 0.70 | ❌ (true: Hostile) |
| _[correct Game Analysis example — fill confidence from notebook]_ | Game Analysis | _0.__ | ✅ |
| _[correct Hostile Trash Talk example — fill confidence from notebook]_ | Hostile Trash Talk | _0.__ | ✅ |

> **TODO before submitting:** re-run the inference cell on 1–2 correctly-predicted test comments to capture their real confidence scores and replace the two placeholder rows. (My logs only saved confidences for the misclassified examples.)

*Why a correct prediction is reasonable:* a comment like "The Spurs ran the same set every possession and the Knicks finally adjusted their pick-and-roll coverage" is predicted **Game Analysis** with high confidence because it cites specific, verifiable tactical detail and carries no emotional or insulting framing — exactly the signal the label is meant to capture.

---

## 7. Reflection — What the Model Learned vs. What I Intended

I intended the model to classify by **intent**: is the comment informing, feeling, or attacking? What it actually learned is closer to **surface tone** — all-caps, exclamation marks, profanity, and comment length. Those features correlate with my labels often enough to reach 0.815, but the correlation breaks exactly where intent and tone diverge:

- It **overfit to form**: aggressive punctuation → Hostile/Visceral, even when the comment is harmless shock (#1) or a genuine tactical point phrased bluntly (#3).
- It **missed quiet hostility**: a short, calm-looking insult ("Suck a fat one, Wemby!") got read as hype because it lacks the loud surface markers the model associates with hostility.

So the model captured *register* well and *intent* only partially. The clean Visceral and Hostile cases (loud + targeted) it nails; the gap is the cases where tone and intent point in different directions — which is precisely the boundary I called out as hardest before training. Closing it needs more counter-example data, not a different model.

---

## 8. Spec Reflection

**One way the spec helped:** the milestone structure forced me to **run the baseline before fine-tuning**. Locking in the 0.542 zero-shot number first made my fine-tuned 0.815 meaningful — without it I'd have had a number with nothing to compare against, and "0.815" alone says little about how hard the task is.

**One way I diverged:** the notebook's default was 3 epochs, but the spec also told me to *read the validation curve and justify hyperparameter choices*. The validation accuracy at 3 epochs (0.59) showed clear underfitting, so I diverged from the default — I raised `num_train_epochs` to 20 and turned on `load_best_model_at_end` so the best validation checkpoint (epoch 10, val acc 0.852) is restored automatically instead of the overfit final weights. The spec's guidance ("a fine-tuned model should meaningfully exceed the baseline") is what prompted me to question the default rather than accept an underfit model.

---

## 9. AI Usage

1. **Label stress-testing (before annotation).** I gave an LLM my three label definitions and the Visceral-vs-Hostile boundary rule and asked it to generate 8 comments that sit on the boundary. Several were ambiguous enough that I couldn't classify them cleanly, which is what pushed me to write the explicit "intent over tone, default to Visceral when torn" rule *before* labeling 178 examples. I overrode the AI's implicit assumption that profanity = hostility — my final rule treats profanity as neutral evidence.

2. **Failure-pattern analysis (after evaluation).** I pasted the 5 misclassified examples into an LLM and asked it to find common themes. It proposed that errors cluster on short comments and on tone/intent mismatch. I verified this against the source comments and kept the tone-vs-intent pattern (it held up across #1, #3, #5), but I **discarded** its claim that errors correlated with comment length — with only 5 errors that pattern wasn't supportable, so I left it out of the report.

*Annotation disclosure:* all 178 final labels were applied by me by hand. The raw `sentiment_label` field in `dataset.csv` came from Reddit/scraping and was used only to sample a varied mix of comments — it was never used as a label.

---

## Repository Contents

| File | Purpose |
|---|---|
| `planning.md` | Pre-project design: labels, edge cases, metrics, AI tool plan |
| `balanced_200_dataset.csv` | Final hand-labeled dataset (`text`, `label`, `notes`) |
| `dataset.csv` | Raw scraped r/nba comments (sampling source) |
| `evaluation_results.json` | Accuracy numbers + label map exported from Colab |
| `confusion_matrix.png` | Fine-tuned confusion matrix (image) |