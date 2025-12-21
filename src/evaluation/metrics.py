import re
from sentence_transformers import SentenceTransformer, util

# Load the model only once
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """Normalizes text by lowercasing, removing punctuation, and stripping whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def semantic_match(generated, expected, thresh=0.75):
    """
    Checks for semantic similarity between generated and expected text.
    Returns a boolean and the similarity score.
    """
    g = preprocess_text(generated)
    e = preprocess_text(expected)
    if not g or not e:
        return False, 0.0
    
    # The model.encode function can take a string or a list of strings.
    # We pass single strings here.
    sim = util.cos_sim(model.encode(g), model.encode(e)).item()
    return sim >= thresh, sim

def compute_metrics(generated, expected):
    """
    Computes a dictionary of robust metrics comparing generated and expected text.
    """
    processed_gen = preprocess_text(generated)
    processed_exp = preprocess_text(expected)
    
    is_sem_match, sem_score = semantic_match(generated, expected)
    
    return {
        "exact_match": processed_gen == processed_exp,
        "semantic_match": is_sem_match,
        "semantic_score": sem_score,
        "generated_processed": processed_gen,
        "expected_processed": processed_exp
    }