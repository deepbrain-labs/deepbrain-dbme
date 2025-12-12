import argparse
import json
import random
import datetime
from typing import List, Dict

import torch
from torch.utils.data import Dataset

class SyntheticData(Dataset):
    """Custom Dataset for synthetic session data."""
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sessions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                session_data = json.loads(line)
                # Concatenate all utterances in a session to form a long context
                long_context = " ".join(session_data['utterances'])
                self.sessions.append(long_context)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session_text = self.sessions[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            session_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # .squeeze(0) to remove the batch dimension pytorch adds
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def generate_session(session_id: str, start_time: datetime.datetime) -> Dict:
    topics = ["weather", "food", "movies", "travel", "hobbies"]
    facts_template = [
        "I like {topic}.",
        "My favorite {topic} is {item}.",
        "I recently went to {location}.",
        "I have a pet {animal}."
    ]
    
    # Simple predefined items for templates
    topic_items = {
        "weather": ["sunny", "rainy", "snowy"],
        "food": ["pizza", "sushi", "tacos"],
        "movies": ["Inception", "The Matrix", "Interstellar"],
        "travel": ["Paris", "Tokyo", "New York"],
        "hobbies": ["reading", "gaming", "hiking"],
        "location": ["the park", "the beach", "the mountains"],
        "animal": ["cat", "dog", "hamster"]
    }

    session_data = {
        "session_id": session_id,
        "timestamp": start_time.isoformat(),
        "utterances": [],
        "facts": []
    }

    current_topic = random.choice(topics)
    
    # Generate 5-10 turns
    num_turns = random.randint(5, 10)
    current_time = start_time

    for i in range(num_turns):
        # User utterance
        if i == 0:
            user_utt = f"Hi, let's talk about {current_topic}."
        elif i % 3 == 0:
             # Inject a fact
            fact_type = random.choice(["item", "location", "animal"]) if current_topic not in topic_items else "item"
            
            if current_topic in topic_items:
                 item = random.choice(topic_items[current_topic])
                 user_utt = f"I really enjoy {current_topic}, especially {item}."
                 session_data["facts"].append({
                     "text": f"User enjoys {current_topic} ({item}).",
                     "timestamp": current_time.isoformat()
                 })
            else:
                 user_utt = f"Tell me more about {current_topic}."
        else:
            user_utt = f"That's interesting. What else?"

        session_data["utterances"].append(f"user: {user_utt}")
        
        # System response (dummy)
        sys_utt = f"I see. {current_topic} is a great topic."
        session_data["utterances"].append(f"system: {sys_utt}")
        
        current_time += datetime.timedelta(seconds=random.randint(5, 60))

    return session_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic episodic sessions.")
    parser.add_argument("--output", type=str, default="data/synthetic_sessions.jsonl", help="Output file path.")
    parser.add_argument("--num_sessions", type=int, default=100, help="Number of sessions to generate.")
    args = parser.parse_args()

    print(f"Generating {args.num_sessions} sessions to {args.output}...")
    
    base_time = datetime.datetime.now()
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for i in range(args.num_sessions):
            # Spread sessions over last 30 days
            session_time = base_time - datetime.timedelta(days=random.randint(0, 30))
            session = generate_session(f"sess_{i:04d}", session_time)
            f.write(json.dumps(session) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()