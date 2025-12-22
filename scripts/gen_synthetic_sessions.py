
import argparse
import json
import random
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument("--n_sessions", type=int, default=500, help="Total sessions")
    parser.add_argument("--n_facts", type=int, default=200, help="Total unique facts to inject")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    
    # 1. Generate Universe of Facts
    # simple kv facts: "The code for {entity} is {number}"
    facts = []
    for i in range(args.n_facts):
        entity = f"entity_{i}"
        val = str(random.randint(1000, 9999))
        facts.append({
            "fact_id": f"f{i}",
            "question": f"What is the code for {entity}?",
            "answer": val,
            "statement": f"The code for {entity} is {val}."
        })

    # 2. Schedule Facts
    # We distribute n_facts across n_sessions (randomly or sequentially).
    # ensure we don't exceed n_sessions.
    # We will randomly pick a session for each fact.
    fact_schedule = {} # session_idx -> list of facts to insert
    
    for f in facts:
        # Pick a random session in the first 80% of generated sessions to allow for retention delays
        # or just random. User wants "detect retention signal". 
        # If we insert at session 499, we can't test delay 50.
        # Let's insert in range [0, n_sessions - 200] if possible, else [0, n_sessions-1]
        max_limit = max(1, args.n_sessions - 201)
        sess_idx = random.randint(0, max_limit)
        if sess_idx not in fact_schedule:
            fact_schedule[sess_idx] = []
        fact_schedule[sess_idx].append(f)

    # 3. Generate Sessions
    sessions = []
    delays = [1, 10, 50, 200]
    
    # Track queries scheduled for future sessions.
    # queries_schedule[sess_idx] = [q1, q2...]
    queries_schedule = {}

    for s_idx in range(args.n_sessions):
        session_obj = {
            "session_id": f"s{s_idx}",
            "step": s_idx,
            "utterances": [],
            "injected_facts": [],
            "queries": [] # Queries to ASK in this session (checking PAST facts)
        }
        
        # A. Insert Distractors (Filler turns)
        distractors_count = 5
        # We'll just put some dummy text
        for _ in range(distractors_count):
            session_obj["utterances"].append(f"user: random talk {random.random():.4f}")
            session_obj["utterances"].append(f"system: interesting {random.random():.4f}")
            
        # B. Insert Facts (if any scheduled)
        if s_idx in fact_schedule:
            turn_idx = 0 # insert at start or random? distinct lines.
            # We insert into utterances
            for f in fact_schedule[s_idx]:
                # Insert as a user statement
                # "By the way, the code for entity_X is 1234."
                u_text = f"user: By the way, {f['statement']}"
                s_text = "system: Got it, saved."
                
                # Append to utterances
                session_obj["utterances"].append(u_text)
                session_obj["utterances"].append(s_text)
                
                # Mark where it happened (approx turn index = len/2 since python list includes user+sys)
                # actually 'time' usually refers to utterance index.
                # Let's say explicit fact_id link.
                
                f_meta = {
                    "fact_id": f['fact_id'],
                    "statement": f['statement'],
                    "turn_index": len(session_obj["utterances"]) - 2 # point to user msg
                }
                session_obj["injected_facts"].append(f_meta)
                
                # Schedule Queries
                for d in delays:
                    target_sess = s_idx + d
                    if target_sess < args.n_sessions:
                        if target_sess not in queries_schedule:
                            queries_schedule[target_sess] = []
                        
                        q_obj = {
                            "fact_id": f['fact_id'],
                            "query_text": f['question'],
                            "expected_answer": f['answer'],
                            "delay": d,
                            "trigger_session_id": target_sess,
                            "origin_session_id": s_idx
                        }
                        queries_schedule[target_sess].append(q_obj)

        # C. Add Scheduled Queries for THIS session
        if s_idx in queries_schedule:
            session_obj["queries"] = queries_schedule[s_idx]
            
        sessions.append(session_obj)

    # 4. Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(sessions, f, indent=2)
        
    print(f"Generated {len(sessions)} sessions with {args.n_facts} unique facts.")
    print(f"Total queries scheduled: {sum(len(v) for v in queries_schedule.values())}")

if __name__ == "__main__":
    main()