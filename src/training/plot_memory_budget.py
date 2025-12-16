import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_memory_vs_accuracy(results_files: list, output_dir: str):
    """
    Plots the final accuracy against the total memory budget.
    
    Args:
        results_files: A list of paths to the metrics.jsonl files.
        output_dir: The directory to save the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    data_points = []

    for file_path in results_files:
        try:
            df = pd.read_json(path_or_buf=file_path, lines=True)
            if df.empty:
                print(f"Warning: {file_path} is empty. Skipping.")
                continue

            # Assuming slot_dim is constant and can be inferred or set
            slot_dim = 256 # You might need to pass this from config
            
            # Final memory is the last recorded size
            final_es_size = df['es_size'].iloc[-1]
            final_kstore_size = df['kstore_size'].iloc[-1]
            total_slots = final_es_size + final_kstore_size
            
            # Calculate memory in KB (assuming float32)
            memory_kb = (total_slots * slot_dim * 4) / 1024
            
            # Overall accuracy
            accuracy = df['is_correct'].mean()
            
            label = os.path.basename(file_path).replace('metrics_', '').replace('.jsonl', '')
            data_points.append({'label': label, 'memory': memory_kb, 'accuracy': accuracy})

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Plotting
    for point in data_points:
        ax.scatter(point['memory'], point['accuracy'], s=150, label=point['label'], alpha=0.7, edgecolors='w')
        ax.text(point['memory'] * 1.05, point['accuracy'], point['label'], fontsize=12)

    ax.set_title('Accuracy vs. Memory Budget', fontsize=18, fontweight='bold')
    ax.set_xlabel('Memory Budget (KB)', fontsize=14)
    ax.set_ylabel('Overall Recall@1 (Accuracy)', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1.0)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'accuracy_vs_memory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs. memory budget from metrics files.")
    parser.add_argument('--results_files', nargs='+', required=True,
                        help="Paths to one or more metrics JSONL files.")
    parser.add_argument('--output_dir', type=str, default='demos/plots',
                        help="Directory to save the plots.")
    args = parser.parse_args()

    plot_memory_vs_accuracy(args.results_files, args.output_dir)

if __name__ == "__main__":
    main()
