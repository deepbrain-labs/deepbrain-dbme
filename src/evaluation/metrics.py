import string
import re
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_MODEL = None
except ImportError:
    _SBERT_MODEL = None
    print("Warning: sentence-transformers not found. Semantic metrics will be disabled.")

def get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            _SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            return None
    return _SBERT_MODEL

def normalize_text(text: str) -> str:
    """
    Normalize text by removing prompt artifacts, punctuation, and lowercasing.
    """
    # Remove "System:", "User:", etc.
    text = re.sub(r'System:|User:', '', text, flags=re.IGNORECASE)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Strip whitespace
    return text.strip()

def compute_metrics(generated: str, expected: str, semantic_threshold: float = 0.8) -> dict:
    """
    Compute Exact Match and Semantic Match metrics.
    """
    norm_gen = normalize_text(generated)
    norm_exp = normalize_text(expected)
    
    # Exact Match (Normalized)
    em = norm_exp in norm_gen # "in" check is looser than ==, good for QA
    
    # Semantic Match
    semantic_match = False
    score = 0.0
    
    model = get_sbert_model()
    if model:
        embeddings = model.encode([generated, expected])
        from sklearn.metrics.pairwise import cosine_similarity
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        if score > semantic_threshold:
            semantic_match = True
    else:
        # Fallback if SBERT not available
        semantic_match = em
        score = 1.0 if em else 0.0
        
    return {
        "exact_match": em,
        "semantic_match": semantic_match,
        "semantic_score": float(score)
    }
