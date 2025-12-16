import argparse
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def plot_slot_clusters(data_path, output_path, method='tsne', n_components=2, perplexity=30.0, max_iter=1000, learning_rate='auto'):
    """
    Loads extracted memory data, performs dimensionality reduction, and plots the slot clusters.
    
    Args:
        data_path (str): Path to the extracted memory data file.
        output_path (str): Path to save the output plot.
        method (str): Dimensionality reduction method ('tsne' or 'pca').
        n_components (int): Number of dimensions for the embedding.
        perplexity (float): Perplexity for t-SNE.
        max_iter (int): Number of iterations for t-SNE.
        learning_rate (float or 'auto'): Learning rate for t-SNE.
    """
    print(f"Loading memory data from {data_path}...")
    with open(data_path, 'rb') as f:
        memory_data = pickle.load(f)
        
    es_slots = memory_data['episodic_store']['slots']
    ks_slots = memory_data['k_store']['slots']
    
    # Ensure there's data to plot
    if es_slots.size == 0 and ks_slots.size == 0:
        print("No memory data to plot.")
        return

    all_slots = np.vstack([es_slots, ks_slots])
    
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=min(perplexity, all_slots.shape[0] - 1), 
                       max_iter=max_iter, learning_rate=learning_rate, random_state=42, init='pca', n_jobs=-1)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 'tsne' or 'pca'.")
        
    embedded_slots = reducer.fit_transform(all_slots)
    
    es_embedded = embedded_slots[:len(es_slots)]
    ks_embedded = embedded_slots[len(es_slots):]
    
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    if es_embedded.shape[0] > 0:
        plt.scatter(es_embedded[:, 0], es_embedded[:, 1], c='blue', alpha=0.5, label='Episodic Store')
    if ks_embedded.shape[0] > 0:
        plt.scatter(ks_embedded[:, 0], ks_embedded[:, 1], c='red', marker='x', s=100, label='K-Store (Prototypes)')
        
    plt.title('t-SNE Visualization of Memory Slot Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    
    print(f"Saving plot to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print("Plot generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot memory slot clusters using t-SNE or PCA.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the extracted memory data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output plot.")
    parser.add_argument("--method", type=str, default='tsne', choices=['tsne', 'pca'], help="Dimensionality reduction method.")
    
    args = parser.parse_args()
    
    plot_slot_clusters(args.data_path, args.output_path, method=args.method)