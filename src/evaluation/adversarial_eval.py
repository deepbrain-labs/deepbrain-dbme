import torch
from src.storage.episodic_store import EpisodicStore

class AdversarialEvaluator:
    def __init__(self, model, tokenizer, es: EpisodicStore):
        self.model = model
        self.tokenizer = tokenizer
        self.es = es

    def inject_stale_fact(self, fact_text, stale_answer):
        """Injects a stale fact into the episodic store."""
        # This is a simplified example. In a real scenario, you would
        # want to create a more realistic context for the fact.
        context = f"A stale fact: {fact_text} is {stale_answer}."
        inputs = self.tokenizer(context, return_tensors="pt")
        
        # We need to get the model's representation for this context
        # to create a memory slot.
        with torch.no_grad():
            _, embeddings = self.model(inputs.input_ids)
            
        # For simplicity, we'll use the last token's embedding to represent the context.
        context_embedding = embeddings[:, -1, :]
        
        # Create a memory slot using the hippocampal encoder.
        key, slot, _ = self.model.he.write(context_embedding)
        
        # Add the stale fact to the episodic store.
        self.es.add(key.unsqueeze(0), slot.unsqueeze(0), meta={"fact": fact_text, "answer": stale_answer, "is_stale": True})

    def evaluate_hallucination(self, fact_text, correct_answer):
        """Evaluates the model's tendency to hallucinate the stale fact."""
        # Ask a question about the fact.
        question = f"What is {fact_text}?"
        inputs = self.tokenizer(question, return_tensors="pt")

        # Generate an answer from the model.
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids)
            
        # Decode the answer.
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check if the answer contains the stale information.
        if correct_answer in answer:
            return "correct"
        
        # Create a query embedding for the fact.
        with torch.no_grad():
            _, fact_embedding = self.model(self.tokenizer(fact_text, return_tensors="pt").input_ids)
            query_key, _, _ = self.model.he.write(fact_embedding.mean(dim=1))

        if self.es.query_by_key(query_key):
            # A more sophisticated check would be needed here to see if the
            # stale answer is in the generated text.
            return "hallucinated"
        else:
            return "unknown"