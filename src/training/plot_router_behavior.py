import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_router_behavior(results_file: str, output_dir: str, window_size: int = 50):
    """
    Plots the router's behavior over time.
    
    Args:
        results_file: Path to the metrics.jsonl file for a 'learned' router run.
        output_dir: The directory to save the plot.
        window_size: The size of the rolling window for averaging.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    try:
        df = pd.read_json(path_or_buf=results_file, lines=True)
        if df.empty:
            print(f"Error: {results_file} is empty.")
            return

        # Convert router choice to numeric for rolling average
        df['choice_is_kstore'] = (df['router_choice'] == 'KStore').astype(int)
        
        # Calculate rolling average of routing to KStore
        df['kstore_routing_frac'] = df['choice_is_kstore'].rolling(window=window_size).mean()
        df['es_routing_frac'] = 1 - df['kstore_routing_frac']

        ax.stackplot(df['event_idx'], df['es_routing_frac'], df['kstore_routing_frac'],
                     labels=['Episodic Store', 'K-Store'],
                     colors=['#3498db', '#e74c3c'],
                     alpha=0.7)

    except Exception as e:
        print(f"Error processing file {results_file}: {e}")
        return

    ax.set_title('Router Behavior Over Time', fontsize=18, fontweight='bold')
    ax.set_xlabel('Event Index (Time)', fontsize=14)
    ax.set_ylabel(f'Fraction of Queries Routed (window={window_size})', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, df['event_idx'].max())
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'router_behavior.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot router behavior from a metrics file.")
    parser.add_argument('--results_file', type=str, required=True,
                        help="Path to a single metrics JSONL file from a learned router experiment.")
    parser.add_argument('--output_dir', type=str, default='demos/plots',
                        help="Directory to save the plots.")
    parser.add_argument('--window', type=int, default=100,
                        help="Size of the rolling window for smoothing.")
    args = parser.parse_args()

    plot_router_behavior(args.results_file, args.output_dir, args.window)

if __name__ == "__main__":
    main()
