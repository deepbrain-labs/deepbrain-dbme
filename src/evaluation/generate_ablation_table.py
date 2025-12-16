import argparse
import json
import pandas as pd
import numpy as np
import os

def generate_ablation_table(results_dir: str, output_file: str, slot_dim: int):
    """
    Generates a Markdown table summarizing the results of the ablation studies.

    Args:
        results_dir: The directory containing all the metrics.jsonl files.
        output_file: The path to save the output Markdown file.
        slot_dim: The dimension of the memory slots.
    """
    
    all_results = []
    
    for filename in os.listdir(results_dir):
        if filename.startswith('metrics_') and filename.endswith('.jsonl'):
            file_path = os.path.join(results_dir, filename)
            
            # Extract variant name and seed from filename
            parts = filename.replace('metrics_', '').replace('.jsonl', '').split('_seed')
            variant = parts[0]
            seed = int(parts[1]) if len(parts) > 1 else 0
            
            try:
                df = pd.read_json(path_or_buf=file_path, lines=True)
                if df.empty:
                    continue
                
                recall = df['is_correct'].mean()
                
                # Memory calculation
                final_es_slots = df['es_size'].iloc[-1]
                final_kstore_slots = df['kstore_size'].iloc[-1]
                memory_kb = ((final_es_slots + final_kstore_slots) * slot_dim * 4) / 1024
                
                avg_latency_ms = df['latency_ms'].mean()
                
                all_results.append({
                    'Variant': variant,
                    'Seed': seed,
                    'Recall@1': recall,
                    'Memory (KB)': memory_kb,
                    'Latency (ms)': avg_latency_ms
                })

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    if not all_results:
        print("No results found to generate a table.")
        return

    # Create a DataFrame and calculate stats
    results_df = pd.DataFrame(all_results)
    
    # Group by variant and calculate mean and std dev
    summary = results_df.groupby('Variant').agg(
        Recall_Mean=('Recall@1', 'mean'),
        Recall_Std=('Recall@1', 'std'),
        Memory_Mean=('Memory (KB)', 'mean'),
        Memory_Std=('Memory (KB)', 'std'),
        Latency_Mean=('Latency (ms)', 'mean'),
        Latency_Std=('Latency (ms)', 'std')
    ).reset_index()

    # Format the table
    summary['Recall@1'] = summary.apply(lambda row: f"{row['Recall_Mean']:.3f} ± {row['Recall_Std']:.3f}", axis=1)
    summary['Memory (KB)'] = summary.apply(lambda row: f"{row['Memory_Mean']:.1f} ± {row['Memory_Std']:.1f}", axis=1)
    summary['Latency (ms)'] = summary.apply(lambda row: f"{row['Latency_Mean']:.1f} ± {row['Latency_Std']:.1f}", axis=1)

    # Select and rename columns for the final table
    final_table = summary[['Variant', 'Recall@1', 'Memory (KB)', 'Latency (ms)']]

    # Convert to Markdown
    markdown_table = final_table.to_markdown(index=False)
    
    # Save the table
    with open(output_file, 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(markdown_table)
        
    print(f"Ablation table saved to {output_file}")
    print("\n" + markdown_table)


def main():
    parser = argparse.ArgumentParser(description="Generate an ablation summary table from metrics files.")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Directory containing all metrics JSONL files from experiments.")
    parser.add_argument('--output_file', type=str, default='demos/ablation_results.md',
                        help="Path to save the Markdown output file.")
    parser.add_argument('--slot_dim', type=int, default=256,
                        help="The dimension of the memory slots.")
    args = parser.parse_args()

    generate_ablation_table(args.results_dir, args.output_file, args.slot_dim)

if __name__ == "__main__":
    main()