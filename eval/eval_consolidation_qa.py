import json
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import compute_metrics

def analyze_consolidation(results_file="results/c2_consolidation.json"):
    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return

    with open(results_file) as f:
        data = json.load(f)

    # Apply robust scoring to each record
    for record in data:
        if "generated" in record and "expected" in record:
            metrics = compute_metrics(record["generated"], record["expected"])
            record.update(metrics)

    df = pd.DataFrame(data)

    # Separate pre- and post-consolidation results
    pre_df = df[df["phase"] == "pre_consolidation"]
    post_df = df[df["phase"] == "post_consolidation"]

    # --- Pre-Consolidation Analysis ---
    if not pre_df.empty:
        pre_exact_acc = pre_df["exact_match"].mean()
        pre_sem_acc = pre_df["semantic_match"].mean()
        print("\n--- Pre-Consolidation QA ---")
        print(f"  Exact Match Accuracy:      {pre_exact_acc:.2%}")
        print(f"  Semantic Match Accuracy:   {pre_sem_acc:.2%}")
    else:
        print("\n--- No Pre-Consolidation data found ---")

    # --- Post-Consolidation Analysis ---
    if not post_df.empty:
        print("\n--- Post-Consolidation QA (by Config) ---")
        
        # Group by the consolidation configuration used
        for config_name, group in post_df.groupby("config"):
            exact_acc = group["exact_match"].mean()
            sem_acc = group["semantic_match"].mean()
            
            print(f"\n  Config: {config_name}")
            print(f"    Exact Match Accuracy:    {exact_acc:.2%}")
            print(f"    Semantic Match Accuracy: {sem_acc:.2%}")
            
            # Show a few examples of mismatches
            mismatches = group[group["exact_match"] == False]
            if not mismatches.empty:
                print("    Mismatch Examples:")
                for i, row in mismatches.head(2).iterrows():
                    print(f"      - Q: {row['query']}")
                    print(f"        Gen: '{row['generated']}' | Exp: '{row['expected']}' | Sem_Score: {row['semantic_score']:.2f}")

    else:
        print("\n--- No Post-Consolidation data found ---")


if __name__ == "__main__":
    analyze_consolidation()