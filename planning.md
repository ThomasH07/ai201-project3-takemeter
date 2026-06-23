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

Handling it during annotation: I will establish a rule based on the intent of the hostility. If the comment is an immediate, frustrated outburst directed at a specific play or general team failure (venting), it will be labeled Visceral Reaction. If the comment contains ad hominem attacks meant to demean a player's general character, cross lines into personal insults, or attack a fanbase (malice), it will be labeled Hostile Trash Talk. When in doubt, annotators will default to Visceral Reaction for heat-of-the-moment venting.

Data Collection Plan
Collection Strategy: I will use the dataset contained in dataset.csv, which consists of 955 actual Reddit comments specifically pared down to two columns: body (the comment text) and sentiment_label.

The CSV already includes sentiment groupings (379 negative, 339 neutral, and 155 positive comments). I will pull a random subset of roughly comments across these different sentiment buckets to train the initial classifier. I aim to collect a balanced distribution of approximately 66-67 examples per label. Because extreme Hostile Trash Talk might still be underrepresented in the 200-comment sample, I will prompt a large language model to generate synthetic edge-case examples that mimic the vernacular of the real comments in the CSV's body column to balance the training data if needed.

Evaluation Metrics
Metrics Used: Precision, Recall, and Macro F1-Score.

In a live game thread, classes are heavily imbalanced. "Visceral Reaction" comments vastly outnumber the other two categories, especially during a historic 29-point comeback. If 80% of the thread is just people screaming in all-caps, a model that simply guesses "Visceral Reaction" every single time will achieve 80% accuracy while failing completely at its actual task. F1-score balances precision (how many flagged toxic comments were actually toxic?) and recall (did we catch all the analytical comments?), giving a true picture of the model's performance across all categories.

Definition of Success
Target Performance: A Macro F1-score of 0.85 or higher.

Real-world Deployment: For this classifier to be genuinely useful for a community tool — such as an r/nba browser extension that filters game threads into a "pure basketball discussion" view or an auto-moderation bot — it needs high precision on the Hostile Trash Talk label (so users aren't unfairly banned) and high recall on the Game Analysis label (so the filtered feed actually captures the smart basketball talk). Achieving an 85% F1-score proves the model has successfully learned the subtle linguistic differences between passionate sports fandom and actual toxicity.

Review of Evaluation Plan:
My success criteria are highly specific and objective. Because I am targeting a strict mathematical threshold (Macro F1-score of 0.85) alongside specific precision/recall goals for individual classes, I can objectively calculate these metrics against my holdout test set to determine exactly whether I hit my deployment goals, eliminating subjective guesswork.

AI Tool Plan

Label Stress-Testing: Before I begin annotating my 200 examples, I will provide my exact label definitions and hard edge case rules to an LLM. I will ask it to generate 5-10 synthetic comments that sit right on the boundary between labels. If I cannot cleanly classify the AI's generated posts using my rubric, I will tighten my definitions before I start labeling.

Annotation Assistance: I will use an LLM (such as Llama-3 or Gemini) to pre-label the batch of 200 examples. I will track this by keeping an ai_pre_label column in my dataset. I will then review every row and apply my final determination in a separate human_label column, ensuring full transparency of what the AI suggested versus what I finalized.

Failure Analysis: After running my model evaluations, I will export a list of all wrong predictions (false positives and false negatives). I will give this list to an AI tool and prompt it to identify structural or linguistic patterns in the misclassifications based on my label definitions. I will then manually review the source comments to verify these patterns, looking for things the model systematically missed, such as heavy sarcasm, rhetorical questions, or team-specific slang.