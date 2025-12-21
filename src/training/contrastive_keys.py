import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.hippocampal_encoder import HippocampalEncoder
from data.create_episodic_qa import generate_qa_sessions

class ContrastiveQADataset(Dataset):
    """
    Creates pairs of (query, correct_fact) for contrastive learning.
    Each item is a dictionary {'query': str, 'positive': str}.
    """
    def __init__(self, qa_sessions):
        self.data = []
        for session in qa_sessions:
            for event in session['events']:
                if event['type'] == 'qa':
                    self.data.append({
                        'query': event['query'],
                        'positive': event['fact']
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def info_nce_loss(query, positive, negatives, temperature=0.1):
    """
    Calculates the InfoNCE loss for a single query.
    query: [D]
    positive: [D]
    negatives: [N, D]
    """
    # Cosine similarity
    sim_pos = F.cosine_similarity(query.unsqueeze(0), positive.unsqueeze(0))
    sim_negs = F.cosine_similarity(query.unsqueeze(0), negatives)
    
    # Numerator and denominator
    numerator = torch.exp(sim_pos / temperature)
    denominator = numerator + torch.sum(torch.exp(sim_negs / temperature))
    
    loss = -torch.log(numerator / denominator)
    return loss

def train_contrastive(encoder, dataloader, optimizer, device):
    encoder.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        queries = batch['query']
        positives = batch['positive']
        
        optimizer.zero_grad()
        
        # In-batch negatives: use all other positives in the batch as negatives
        batch_loss = 0
        
        # This is a simplified implementation. A more efficient way would be to
        # compute all embeddings once and then use matrix operations.
        for i in range(len(queries)):
            query_text = queries[i]
            positive_text = positives[i]
            
            # Create a list of negative texts for the current query
            negative_texts = [p for j, p in enumerate(positives) if i != j]
            
            # This part is slow due to repeated encoding. For a real implementation,
            # you'd pre-tokenize or use a model that handles raw strings.
            # For this script, we'll simulate this with placeholder embeddings.
            # A real implementation requires a text encoder (e.g., from the LM).
            # We will use random tensors to simulate the output of a text encoder.
            query_emb = torch.randn(1, 768).to(device)
            positive_emb = torch.randn(1, 768).to(device)
            negative_embs = torch.randn(len(negative_texts), 768).to(device)

            # Pass through the hippocampal encoder to get keys
            query_key, _, _ = encoder(query_emb)
            positive_key, _, _ = encoder(positive_emb)
            negative_keys, _, _ = encoder(negative_embs)
            
            loss = info_nce_loss(query_key.squeeze(0), positive_key.squeeze(0), negative_keys)
            batch_loss += loss

        batch_loss /= len(queries) # Average loss for the batch
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train Hippocampal Encoder with contrastive loss.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--n_sessions", type=int, default=1000, help="Number of QA sessions to generate for training data.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate or load data
    print(f"Generating {args.n_sessions} QA sessions for training data...")
    qa_sessions = generate_qa_sessions(args.n_sessions)
    dataset = ContrastiveQADataset(qa_sessions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize model and optimizer
    # Dimensions must match the HippocampalEncoder's expected input
    encoder = HippocampalEncoder(input_dim=768, slot_dim=256, key_dim=128).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    
    # 3. Training loop
    print("\nStarting contrastive training...")
    for epoch in range(args.epochs):
        avg_loss = train_contrastive(encoder, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    # 4. Save the trained model
    output_path = "models/he_contrastive.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(encoder.state_dict(), output_path)
    print(f"\nTrained encoder saved to {output_path}")

if __name__ == "__main__":
    main()
