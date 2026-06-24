import pandas as pd
import re

# ─── 1. CORE CLASSIFICATION LOGIC ─────────────────────────────────────────
def classify_comment(text):
    """Classifies a single text string into one of three specific r/nba categories."""
    if not isinstance(text, str):
        return "Visceral Reaction"
        
    text_lower = text.lower()
    
    insults = [
        r'fuck\w*', 'bitch', 'loser', 'ass', 'idiot', 'moron', 'clown', 'trash', 
        'poverty', 'fraud', 'choke', 'suck', 'dumb', r'shit\w*', 'garbage', 
        'dogshit', 'weak', 'washed', 'bum', 'scrub', 'soft', 'nephew', 
        'casual', 'delusional', 'cope', 'seethe', 'crybaby', 'flop', 
        'flopper', 'rigged', 'overrated', 'pathetic', 'stfu', 'bullshit',
        'joke', 'useless', 'embarrassing', 'meatrider', 'glazer', 'bandwagon'
    ]
    
    analysis_terms = [
        'foul', 'defense', 'offense', 'coach', 'pass', 'rebound', 'turnover', 
        'quarter', 'half', 'minutes', 'spacing', 'shift', 'trap', 'ref', 
        'call', 'stats', 'adjustment', 'tactic', 'rotation', 'possession', 
        'screen', 'pick', 'roll', 'isolation', 'iso', 'paint', 'perimeter', 
        'zone', 'switch', 'closeout', 'efficiency', 'scheme', 'execution', 
        'timeout', 'lineup', 'matchup', 'whistle', 'assist', 'block', 'steal', 
        'midrange', 'transition', 'bench', 'starters', 'boards', 'fg%', 'ts%'
    ]
    
    has_insult = any(re.search(rf'\b{word}\b', text_lower) for word in insults)
    has_analysis = any(re.search(rf'\b{word}\b', text_lower) for word in analysis_terms)
    is_shouting = text.isupper() and len(text) > 10
    
    if has_insult:
        return "Hostile Trash Talk"
    elif has_analysis and len(text) > 40 and not is_shouting:
        return "Game Analysis"
    else:
        return "Visceral Reaction"


# ─── 2. DATASET BALANCING LOGIC ───────────────────────────────────────────
def create_balanced_dataset(df, target_per_class=67):
    """Samples the dataframe to ensure an equal number of rows per label."""
    balanced_df = pd.DataFrame()

    for label in ['Game Analysis', 'Hostile Trash Talk', 'Visceral Reaction']:
        class_subset = df[df['label'] == label]
        
        # Take target_per_class, or whatever is available if it's less
        sample_size = min(len(class_subset), target_per_class)
        sampled = class_subset.sample(n=sample_size, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled])

    # Shuffle the combined rows so the labels are mixed up
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─── 3. FORMATTING & SAVING ───────────────────────────────────────────────
def format_and_save(df, output_name):
    """Renames columns, adds the 'notes' column, and exports to CSV."""
    # Keep only the required columns and rename 'body' to 'text'
    export_df = df[['body', 'label']].copy()
    export_df.rename(columns={'body': 'text'}, inplace=True)
    
    # Add an empty notes column for manual review
    export_df['notes'] = ""
    
    # Save to CSV
    export_df.to_csv(output_name, index=False)
    return export_df


# ─── 4. MAIN EXECUTION BLOCK ──────────────────────────────────────────────
def main():
    file_name = "dataset.csv"
    
    print("Loading data...")
    df = pd.read_csv(file_name)

    print("Applying classification logic...")
    df['label'] = df['body'].apply(classify_comment)

    # Output 1: The Full Dataset
    full_output_name = "full_classified_" + file_name
    print(f"\n--- Generating Full Dataset ({full_output_name}) ---")
    full_export = format_and_save(df, full_output_name)
    print(f"Total rows: {len(full_export)}")
    print(full_export['label'].value_counts())

    # Output 2: The Balanced Dataset (~250 comments)
    balanced_output_name = "balanced_250_" + file_name
    print(f"\n--- Generating Balanced Dataset ({balanced_output_name}) ---")
    
    #target_per_class set to 84 (84 * 3 classes = 252 rows)
    balanced_df = create_balanced_dataset(df, target_per_class=84) 
    
    balanced_export = format_and_save(balanced_df, balanced_output_name)
    print(f"Total rows: {len(balanced_export)}")
    print(balanced_export['label'].value_counts())
    
    print("\n✅ All processing complete! Both CSVs are ready.")

# This ensures the code only runs if you execute the script directly
if __name__ == "__main__":
    main()