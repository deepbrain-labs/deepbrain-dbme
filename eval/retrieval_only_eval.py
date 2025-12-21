import json
import os
import pandas as pd
import argparse

def load_results(file_path):
    """Loads a JSON results file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    with open(file_path, "r") as f:
        return json.load(f)

def evaluate_retrieval(results_data):
    """
    Computes retrieval metrics (hit@1, hit@5) from results data.
    
    The script assumes the input data contains keys like:
    - 'retrieval_at_1': boolean
    - 'retrieval_at_5': boolean
    - 'delay': integer for grouping
    """
    if not results_data:
        return {}
        
    df = pd.DataFrame(results_data)
    
    if 'retrieval_at_1' not in df.columns or 'retrieval_at_5' not in df.columns:
        print("Error: 'retrieval_at_1' or 'retrieval_at_5' not found in results.")
        return {}

    # Overall Metrics
    overall_hit_at_1 = df['retrieval_at_1'].mean()
    overall_hit_at_5 = df['retrieval_at_5'].mean()
    
    metrics = {
        'overall': {
            'hit@1': overall_hit_at_1,
            'hit@5': overall_hit_at_5,
            'count': len(df)
        }
    }
    
    # Per-Delay Metrics
    if 'delay' in df.columns:
        metrics['by_delay'] = {}
        delays = sorted(df['delay'].unique())
        for d in delays:
            sub_df = df[df['delay'] == d]
            metrics['by_delay'][d] = {
                'hit@1': sub_df['retrieval_at_1'].mean(),
                'hit@5': sub_df['retrieval_at_5'].mean(),
                'count': len(sub_df)
            }
            
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval-only metrics from experiment results.")
    parser.add_argument("--results", type=str, required=True, help="Path to the results JSON file (e.g., results/dbme_retention.json)")
    args = parser.parse_args()
    
    results_data = load_results(args.results)
    
    if results_data:
        retrieval_metrics = evaluate_retrieval(results_data)
        
        if retrieval_metrics:
            print(f"\n--- Retrieval Evaluation for {args.results} ---")
            
            # Overall
            overall = retrieval_metrics['overall']
            print("\nOverall Performance:")
            print(f"  Hit@1 Rate: {overall['hit@1']:.2%}")
            print(f"  Hit@5 Rate: {overall['hit@5']:.2%}")
            print(f"  Total Queries: {overall['count']}")
            
            # By Delay
            if 'by_delay' in retrieval_metrics:
                print("\nPerformance by Delay:")
                print(f"{'Delay':<10} | {'Hit@1':<10} | {'Hit@5':<10} | {'Count':<10}")
                print("-" * 47)
                for delay, m in retrieval_metrics['by_delay'].items():
                    print(f"{delay:<10} | {m['hit@1']:<10.2%} | {m['hit@5']:<10.2%} | {m['count']:<10}")

if __name__ == "__main__":
    main()