import numpy as np
import faiss
import torch
from transformers import GPT2Tokenizer, GPT2Model
from typing import List, Dict, Any

class SimpleRetriever:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # GPT-2 small dimension is 768
        self.dimension = self.model.config.n_embd
        self.index = faiss.IndexFlatIP(self.dimension)
        self.docs = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Computes the average token embedding for the input text.
        This uses the static embedding layer of GPT-2 (fast, baseline method).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            # Get token embeddings from the embedding layer directly
            # shape: (1, seq_len, hidden_dim)
            token_embeddings = self.model.wte(input_ids)
            
            # Average over the sequence length
            # shape: (1, hidden_dim)
            avg_embedding = token_embeddings.mean(dim=1)
            
        return avg_embedding.cpu().numpy()

    def add_documents(self, documents: List[str]):
        """
        Adds a list of documents (strings) to the index.
        """
        if not documents:
            return
            
        embeddings = []
        for doc in documents:
            emb = self._get_embedding(doc)
            embeddings.append(emb)
            self.docs.append(doc)
            
        embeddings_np = np.concatenate(embeddings, axis=0)
        # Normalize for Inner Product if you want Cosine Similarity behavior
        # But here we stick to raw implementation, though usually IP on normalized vectors = Cosine
        faiss.normalize_L2(embeddings_np)
        
        self.index.add(embeddings_np)
        print(f"Added {len(documents)} documents to index. Total: {self.index.ntotal}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for the k nearest neighbors of the query.
        """
        if self.index.ntotal == 0:
            return []
            
        query_emb = self._get_embedding(query)
        faiss.normalize_L2(query_emb)
        
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "text": self.docs[idx],
                    "score": float(distances[0][i]),
                    "id": int(idx)
                })
        
        return results

if __name__ == "__main__":
    # Correctness check
    retriever = SimpleRetriever()
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy hound.",
        "The weather is sunny and bright today.",
        "It is raining cats and dogs outside."
    ]
    retriever.add_documents(texts)
    
    query = "fox jumping dog"
    results = retriever.search(query, k=2)
    
    print(f"\nQuery: '{query}'")
    for res in results:
        print(f"Match: {res['text']} (Score: {res['score']:.4f})")
