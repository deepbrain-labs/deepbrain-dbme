import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def trace_consolidation(data_path, output_path, prototype_id, method='tsne'):
    """
    Visualizes the consolidation of episodic memories into a K-Store prototype.
    
    Args:
        data_path (str): Path to the extracted memory data file.
        output_path (str): Path to save the output plot.
        prototype_id (int): The ID of the prototype to trace.
        method (str): Dimensionality reduction method ('tsne' or 'pca').
    """
    print(f"Loading memory data from {data_path}...")
    with open(data_path, 'rb') as f:
        memory_data = pickle.load(f)
        
    es_slots = memory_data['episodic_store']['slots']
    es_meta = memory_data['episodic_store']['metadata']
    ks_slots = memory_data['k_store']['slots']
    ks_meta = memory_data['k_store']['metadata']
    
    # Find the prototype and its source episodic memories
    try:
        prototype_slot = ks_slots[prototype_id]
        source_ids = ks_meta[prototype_id].get('source_ids', [])
    except IndexError:
        print(f"Prototype with ID {prototype_id} not found.")
        return
        
    if not source_ids:
        print(f"No source IDs found for prototype {prototype_id}.")
        return

    source_slots = []
    for i, meta in enumerate(es_meta):
        if meta.get('id') in source_ids:
            source_slots.append(es_slots[i])

    if not source_slots:
        print("Could not find the source episodic slots.")
        return

    source_slots = np.array(source_slots)
    all_slots_to_plot = np.vstack([source_slots, prototype_slot.reshape(1, -1)])
    
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(10.0, all_slots_to_plot.shape[0] - 1), random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Unsupported method. Choose 'tsne' or 'pca'.")

    embedded_slots = reducer.fit_transform(all_slots_to_plot)
    
    source_embedded = embedded_slots[:-1]
    prototype_embedded = embedded_slots[-1]
    
    print("Generating plot...")
    plt.figure(figsize=(10, 7))
    plt.scatter(source_embedded[:, 0], source_embedded[:, 1], c='blue', alpha=0.7, label='Source Episodic Memories')
    plt.scatter(prototype_embedded[0], prototype_embedded[1], c='red', marker='x', s=150, label=f'Prototype {prototype_id}')
    
    # Draw lines from source memories to the prototype
    for point in source_embedded:
        plt.plot([point[0], prototype_embedded[0]], [point[1], prototype_embedded[1]], 'g--', alpha=0.4)

    plt.title(f'Consolidation Trace for Prototype {prototype_id}')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.legend()
    plt.grid(True)
    
    print(f"Saving plot to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print("Consolidation trace generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace the consolidation of episodic memories into a prototype.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the extracted memory data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output plot.")
    parser.add_argument("--prototype_id", type=int, required=True, help="The ID of the K-Store prototype to trace.")
    parser.add_argument("--method", type=str, default='tsne', choices=['tsne', 'pca'], help="Dimensionality reduction method.")
    
    args = parser.parse_args()
    
    trace_consolidation(args.data_path, args.output_path, args.prototype_id, args.method)