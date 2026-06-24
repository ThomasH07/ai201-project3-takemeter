r/nba (Specifically, Live Game Threads)

Live game threads in r/nba offer a highly concentrated stream of real-time, emotionally charged text data. This specific thread — Game 4 of the 2026 NBA Finals, where the Knicks completed an incredible comeback after being down 29 points against the Spurs — is an absolute goldmine for classification.

The discourse is incredibly varied because the emotional tenor of the thread completely flips. You have analytical breakdowns of player positioning, pure unadulterated despair/hype depending on the fanbase, and highly aggressive trash talk. This massive swing in momentum creates a naturally diverse and linguistically rich dataset.

Labels

1. Game Analysis
This label consists of comments that objectively discuss tactics, player statistics, coaching adjustments, or refereeing without relying primarily on extreme emotion or insults.

"The Spurs put out the worst half offensively that I’ve seen in so long. They looked completely lost. And Wemby gives them nothing in the pose somehow."

"Did KAT deflect the pass? Because that play was going to work for the spurs?"

2. Visceral Reaction
These are highly emotional expressions of shock, joy, despair, or hype. These usually lack deeper analytical substance and are often characterized by all-caps, repetition, or exaggerated punctuation.

"!!! _ have I just witnessed, this shit is insane, I thought this was over with the 2 KAT fouls in minute 1, let aside when they were down by 29 points."

"I PEED MY _ PANTS AND I'M GONNA RUN THROUGH A WALL"

3. Hostile Trash Talk
This category describes comments designed to mock, insult, or demean players, coaches, entire fanbases, or other Reddit users.

" you _ _ losers … bring out all the comments talking smack about Brunson this Brunson that. He made a donkey out of the Texas river walk idiots. Go back to them... women!"

"Spurs with the worst meltdown of all time. Wemby is NOT future GOAT"

Hard Edge Cases
The Ambiguity: The line between Visceral Reaction and Hostile Trash Talk becomes highly ambiguous when a fan is venting aggressively about their own team's staff or mistakes in the heat of the moment. Handling it during annotation: I will establish a rule based on the intent of the hostility. If the comment is an immediate, frustrated outburst directed at a specific play (venting), it will be labeled Visceral Reaction. If it contains ad hominem attacks meant to demean character (malice), it will be labeled Hostile Trash Talk.

Here are three specific difficult examples I encountered and how I decided to label them:

Example: "Brunson is literally throwing this game away into the trash can, I can't watch this garbage anymore."

The Conflict: It uses words like "trash" and "garbage" (Hostile Trash Talk), but it's clearly a fan despairing over a specific game action.

My Decision: Visceral Reaction. It is heat-of-the-moment venting rather than a malicious character attack on Brunson.

Example: "Thibs needs to be investigated by the FBI for terrorism with these rotations."

The Conflict: Contains analytical concepts ("rotations") but pairs them with extreme, hyperbolic insults ("terrorism").

My Decision: Visceral Reaction. The focus is on exaggeration and shock value over actual game analysis or genuine malice.

Example: "Popovich is completely washed, he hasn't known how to coach since Duncan left and the Spurs are a poverty franchise."

The Conflict: It mentions coaching, which usually leans toward Game Analysis, but the tone is incredibly aggressive and insulting.

My Decision: Hostile Trash Talk. Calling the coach "washed" and the team a "poverty franchise" crosses the line from venting into intentional malice and demeaning language.

Handling it during annotation: I will establish a rule based on the intent of the hostility. Because I am using a Python script to pre-label the data using strict keyword heuristics (e.g., any comment with words like "trash" or "poverty" will automatically be flagged as Hostile Trash Talk, and words like "rotations" flagged as Game Analysis), the script will inevitably misclassify these nuanced edge cases. During my manual review phase, I will correct the script's rigid logic. If a comment is an immediate, frustrated outburst directed at a specific play (venting), I will correct it to Visceral Reaction. If it contains intentional malice, I will leave it as Hostile Trash Talk. When in doubt, I will default to Visceral Reaction for heat-of-the-moment venting.

Data Collection Plan
Collection Strategy: I will use the dataset contained in dataset.csv, which consists of 955 actual Reddit comments. I will process this raw data using a custom Python script that pares the dataset down to text, label, and an empty notes column.

Instead of relying on LLM generation for balance, the script will automatically apply regex-based classification logic across the full dataset. It will then dynamically sample exactly 84 comments per category (or the maximum available) to programmatically output a perfectly balanced dataset of roughly 250 comments (balanced_250_dataset.csv). This ensures representation across all classes using only genuine user data. The final dataset will be split 70% for training, 15% for validation, and 15% for testing.

Evaluation Metrics
Metrics Used: Accuracy, Precision, Recall, and Macro F1-Score.

In a live game thread, classes are heavily imbalanced. "Visceral Reaction" comments vastly outnumber the other two categories, especially during a historic 29-point comeback. If 80% of the thread is just people screaming in all-caps, a model that simply guesses "Visceral Reaction" every single time will achieve 80% accuracy while failing completely at its actual task. F1-score balances precision and recall to show true performance. However, because my specific test set will be stratified and artificially balanced by my script, Overall Accuracy will be the primary metric used to directly compare my fine-tuned model's performance against the zero-shot baseline, with F1-score acting as a supplementary metric to ensure no class is left behind.

Definition of Success
Target Performance: An Overall Accuracy and Macro F1-score of 0.85 or higher.

Real-world Deployment: For this classifier to be genuinely useful for a community tool — such as an r/nba browser extension that filters game threads into a "pure basketball discussion" view or an auto-moderation bot — it needs high precision on the Hostile Trash Talk label (so users aren't unfairly banned) and high recall on the Game Analysis label (so the filtered feed actually captures the smart basketball talk). Achieving an 85% accuracy/F1-score proves the model has successfully learned the subtle linguistic differences between passionate sports fandom and actual toxicity.

Review of Evaluation Plan:
I am targeting a strict mathematical threshold (Overall Accuracy of 0.85) alongside specific precision/recall goals for individual classes, I can  calculate these metrics against my holdout test set to determine exactly whether I hit my deployment goals, eliminating subjective guesswork.

AI Tool Plan

Label Stress-Testing: Before I begin annotating my 250 examples, I will provide my exact label definitions and hard edge case rules to an LLM. I will ask it to generate 5-10 synthetic comments that sit right on the boundary between labels. If I cannot cleanly classify the AI's generated posts using my rubric, I will tighten my definitions before I start labeling.

Pre-Labeling and Annotation Assistance: Instead of an LLM, I used a custom Python script (labeling.py) relying on Regular Expressions and predefined word lists to label the batch of 250 examples. The script looks for an insults list to flag Hostile Trash Talk, and an analysis_terms list (combined with string length and capitalization checks) to flag Game Analysis, defaulting everything else to Visceral Reaction.

Note on execution vs. plan: I originally intended to review every row and correct the script's mistakes by hand (logging fixes in the notes column). In practice I did not complete this manual review pass — the labels are the script's raw output, and the notes column is empty. This is disclosed in the README's AI usage section, and the evaluation report analyzes the resulting label noise directly (cases where a single keyword like "coach" or "choke" forces a wrong label). The skipped review is the clearest thing I would change if redoing the project.

Model Training and Baseline Comparison: I will fine-tune a distilbert-base-uncased model on my annotated data for 20 epochs. To establish a baseline, I will use the Groq API to prompt the llama-3.3-70b-versatile model as a zero-shot classifier on the test set. I will then directly compare the overall accuracy of the two models to quantify the improvement achieved by fine-tuning.

Failure Analysis: After running my model evaluations, I will export a list of all wrong predictions (false positives and false negatives). I will give this list to an AI tool and prompt it to identify structural or linguistic patterns in the misclassifications based on my label definitions. I will then manually review the source comments to verify these patterns, looking for things the model systematically missed, such as heavy sarcasm, rhetorical questions, or team-specific slang.