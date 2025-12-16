import os
import pickle
import sys
import numpy as np
import subprocess

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from src.visualization.extract_memory_data import DummyDBME

def run_extraction(model_path, data_path):
    """Runs the data extraction script."""
    print("--- Running Data Extraction ---")
    if not os.path.exists(model_path):
        print(f"Error: Model state file '{model_path}' not found.")
        print("Please run the real data simulation first: python src/training/run_real_episodic_stream.py")
        sys.exit(1)
        
    subprocess.run([
        sys.executable, "src/visualization/extract_memory_data.py",
        "--model_path", model_path,
        "--output_path", data_path
    ], check=True)
    print(f"--- Data Extraction Complete ---")


def run_visualizations(model_path, data_path, output_dir):
    """Runs all the visualization scripts on the generated data."""
    print("\nRunning visualization scripts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot slot clusters
    subprocess.run([
        sys.executable, "src/visualization/plot_slot_clusters.py",
        "--data_path", data_path,
        "--output_path", os.path.join(output_dir, "slot_clusters.png")
    ], check=True)
    
    # 2. Plot memory timeline
    subprocess.run([
        sys.executable, "src/visualization/plot_memory_timeline.py",
        "--data_path", data_path,
        "--output_path", os.path.join(output_dir, "memory_timeline.png")
    ], check=True)

    # 3. Inspect retrieval for "Paris"
    print("\n--- Running Retrieval Inspection ---")
    subprocess.run([
        sys.executable, "src/visualization/inspect_retrieval.py",
        "--model_path", model_path,
        "--query", "What is in Paris?",
    ], check=True)
    print("--- End Retrieval Inspection ---\n")

    # 4. Trace consolidation for the "Paris" prototype (ID 0)
    subprocess.run([
        sys.executable, "src/visualization/trace_consolidation.py",
        "--data_path", data_path,
        "--output_path", os.path.join(output_dir, "consolidation_trace_paris.png"),
        "--prototype_id", "0"
    ], check=True)
    
    print("All visualizations generated successfully.")


if __name__ == "__main__":
    # Point to the output of the real data simulation
    MODEL_PATH = "demos/models/real_data_model_state.pt"
    DATA_PATH = "demos/data/real_memory_data.pkl"
    OUTPUT_DIR = "demos/outputs"
    
    # Step 1: Extract data from the saved model state
    run_extraction(MODEL_PATH, DATA_PATH)
    
    # Step 2: Run visualizations on the extracted data
    run_visualizations(MODEL_PATH, DATA_PATH, OUTPUT_DIR)