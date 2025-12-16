import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_forgetting_curve(results_files: list, output_dir: str, contradicted_fact_query: str, incorrect_answer: str):
    """
    Plots the hallucination rate for a specific contradicted fact over time.

    Args:
        results_files: List of paths to metrics.jsonl files.
        output_dir: Directory to save the plot.
        contradicted_fact_query: The specific query text for the fact that was contradicted.
        incorrect_answer: The incorrect answer for the contradicted fact.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for file_path in results_files:
        try:
            df = pd.read_json(path_or_buf=file_path, lines=True)
            if df.empty:
                print(f"Warning: {file_path} is empty. Skipping.")
                continue

            # Filter for the specific query
            forget_df = df[df['query'] == contradicted_fact_query].copy()
            if forget_df.empty:
                print(f"Warning: Could not find query '{contradicted_fact_query}' in {file_path}. Skipping.")
                continue
            
            # Hallucination is when the generated answer contains the incorrect fact
            forget_df['is_hallucinating'] = forget_df['generated_answer'].str.contains(incorrect_answer, case=False)
            
            # We are plotting a single point for each model, representing the final state
            final_hallucination_rate = forget_df['is_hallucinating'].mean()

            label = os.path.basename(file_path).replace('metrics_', '').replace('.jsonl', '')
            
            ax.bar(label, final_hallucination_rate, label=label, alpha=0.7)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    ax.set_title('Forgetting Stale Facts (Hallucination Rate)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Model Variant', fontsize=14)
    ax.set_ylabel('Hallucination Rate', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'forgetting_hallucination_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot forgetting curve from metrics files.")
    parser.add_argument('--results_files', nargs='+', required=True,
                        help="Paths to metrics JSONL files from the forgetting experiment.")
    parser.add_argument('--output_dir', type=str, default='demos/plots',
                        help="Directory to save the plots.")
    parser.add_argument('--query', type=str, required=True,
                        help="The exact query text of the contradicted fact.")
    parser.add_argument('--incorrect_answer', type=str, required=True,
                        help="The incorrect answer for the contradicted fact.")
    args = parser.parse_args()

    plot_forgetting_curve(args.results_files, args.output_dir, args.query, args.incorrect_answer)

if __name__ == "__main__":
    main()
