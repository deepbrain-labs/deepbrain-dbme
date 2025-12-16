import argparse
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

def plot_memory_timeline(data_path, output_path):
    """
    Loads extracted memory data and plots a timeline of memory events.
    
    Args:
        data_path (str): Path to the extracted memory data file.
        output_path (str): Path to save the output plot.
    """
    print(f"Loading memory data from {data_path}...")
    with open(data_path, 'rb') as f:
        memory_data = pickle.load(f)
        
    es_meta = memory_data['episodic_store']['metadata']
    ks_meta = memory_data['k_store']['metadata']
    
    es_timestamps = [m.get('timestamp', 0) for m in es_meta]
    ks_timestamps = [m.get('consolidation_timestamp', 0) for m in ks_meta]

    # Convert timestamps to a plottable format if they are not already
    # For this example, we'll assume timestamps are simple numerical values (e.g., training steps)
    
    print("Generating timeline plot...")
    plt.figure(figsize=(15, 6))
    
    if es_timestamps:
        plt.scatter(es_timestamps, np.zeros_like(es_timestamps) + 1, c='blue', alpha=0.6, label='Episodic Write')
    if ks_timestamps:
        plt.scatter(ks_timestamps, np.zeros_like(ks_timestamps) + 1.1, c='red', marker='x', s=100, label='Consolidation')
    
    plt.yticks([1, 1.1], ['Episodic Store', 'K-Store'])
    plt.xlabel('Time (Training Step)')
    plt.title('Memory Event Timeline')
    plt.legend(loc='upper left')
    plt.grid(axis='x')
    
    print(f"Saving plot to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print("Timeline generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a timeline of memory events.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the extracted memory data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output plot.")
    
    args = parser.parse_args()
    
    plot_memory_timeline(args.data_path, args.output_path)