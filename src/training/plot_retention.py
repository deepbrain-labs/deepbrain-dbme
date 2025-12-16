import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_retention(results_files: list, output_dir: str, window_size: int = 10):
    """
    Plots the rolling average of recall (accuracy) over time (event index).
    
    Args:
        results_files: A list of paths to the metrics.jsonl files.
        output_dir: The directory to save the plot.
        window_size: The size of the rolling window for averaging.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for file_path in results_files:
        try:
            df = pd.read_json(path_or_buf=file_path, lines=True)
            if df.empty:
                print(f"Warning: {file_path} is empty. Skipping.")
                continue

            # Calculate rolling accuracy
            df['rolling_accuracy'] = df['is_correct'].rolling(window=window_size).mean()
            
            # Get the experiment name from the file path
            label = os.path.basename(file_path).replace('metrics_', '').replace('.jsonl', '')

            ax.plot(df['event_idx'], df['rolling_accuracy'], label=label, linewidth=2)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    ax.set_title('Recall@1 vs. Time (Rolling Average)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Event Index (Time)', fontsize=14)
    ax.set_ylabel(f'Rolling Accuracy (window={window_size})', fontsize=14)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'retention_vs_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot retention curve from metrics files.")
    parser.add_argument('--results_files', nargs='+', required=True,
                        help="Paths to one or more metrics JSONL files.")
    parser.add_argument('--output_dir', type=str, default='demos/plots',
                        help="Directory to save the plots.")
    parser.add_argument('--window', type=int, default=50,
                        help="Size of the rolling window for smoothing the accuracy.")
    args = parser.parse_args()

    plot_retention(args.results_files, args.output_dir, args.window)

if __name__ == "__main__":
    main()
