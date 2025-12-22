
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import glob

def compute_retention_curve(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    # Group by delay
    by_delay = defaultdict(list)
    for r in data:
        # DBME uses 'correct' boolean
        # Baselines use 'correct' boolean
        delay = r['delay']
        correct = 1 if r['correct'] else 0
        by_delay[delay].append(correct)
        
    # Calculate recall @ delay
    delays = sorted(by_delay.keys())
    recalls = [np.mean(by_delay[d]) for d in delays]
    
    return delays, recalls

def compute_aurc(delays, recalls):
    # Area Under Retention Curve
    # We can use trapezoidal rule or just mean if delays are uniform (they are not).
    # Usually we normalize x axis (log or linear).
    # User requested "log x axis" for plot.
    # For AURC, usually linear AUC is standard, or AUC on log-x?
    # "DBME mean AURC >= +15%"
    # Let's compute standard AUC (trapezoidal) using log x? No, standard AUC is linear.
    # But if delays are 1, 10, 50, 200, linear AUC is dominated by the long tail.
    # Let's stick to standard approx (mean recall) or trapezoid.
    # Given the sparse points, let's just do weighted average or raw trapezoid.
    return np.trapz(recalls, delays) / (delays[-1] - delays[0]) if len(delays)>1 else 0

def analyze():
    # Find all result files
    methods = {
        'DBME': glob.glob("results/retention/dbme_seed*.json"),
        'KV': glob.glob("results/retention/kv_seed*.json"),
        'Retrieval': glob.glob("results/retention/retrieval_seed*.json")
    }
    
    plt.figure(figsize=(10, 6))
    
    summary_stats = []
    
    colors = {'DBME': 'blue', 'KV': 'red', 'Retrieval': 'green'}
    
    for method, files in methods.items():
        if not files:
            print(f"No files found for {method}")
            continue
            
        all_delays = None
        all_recalls = []
        aurcs = []
        r1_200 = [] # Recall at delay 200
        
        for f in files:
            d, r = compute_retention_curve(f)
            all_recalls.append(r)
            if all_delays is None:
                all_delays = d
            
            aurc = compute_aurc(d, r)
            aurcs.append(aurc)
            
            # Find delay 200
            try:
                idx = d.index(200)
                r1_200.append(r[idx])
            except ValueError:
                pass
                
        # Mean Curve
        mean_recalls = np.mean(all_recalls, axis=0)
        std_recalls = np.std(all_recalls, axis=0)
        
        # Plot
        plt.plot(all_delays, mean_recalls, label=f"{method} (n={len(files)})", marker='o', color=colors.get(method, 'black'))
        plt.fill_between(all_delays, mean_recalls - std_recalls, mean_recalls + std_recalls, alpha=0.1, color=colors.get(method, 'black'))
        
        # Stats
        mean_aurc = np.mean(aurcs) if aurcs else 0
        mean_r200 = np.mean(r1_200) if r1_200 else 0
        
        summary_stats.append({
            "Method": method,
            "AURC": mean_aurc,
            "R@200": mean_r200,
            "Seeds": len(files)
        })
        
    plt.xscale('log')
    plt.xlabel('Delay (Sessions)')
    plt.ylabel('Recall @ 1')
    plt.title('Retention Curve (Exp A)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/retention_curve.png")
    print("Saved figs/retention_curve.png")
    
    # Print Table
    df = pd.DataFrame(summary_stats)
    print("\n=== Results Summary ===")
    print(df.to_string())
    
    # Acceptance Check
    # DBME mean AURC ≥ +15% vs best baseline
    # Recall@1 at Δ=200: DBME > baseline by ≥0.10
    
    if not df.empty and 'DBME' in df['Method'].values:
        dbme_row = df[df['Method'] == 'DBME'].iloc[0]
        others = df[df['Method'] != 'DBME']
        
        if not others.empty:
            best_baseline_aurc = others['AURC'].max()
            best_baseline_r200 = others['R@200'].max()
            
            aurc_lift = (dbme_row['AURC'] - best_baseline_aurc) / (best_baseline_aurc + 1e-6)
            r200_diff = dbme_row['R@200'] - best_baseline_r200
            
            print(f"\nDBME vs Best Baseline:")
            print(f"AURC Lift: {aurc_lift*100:.2f}% (Target: >= 15%)")
            print(f"R@200 Diff: {r200_diff:.4f} (Target: >= 0.10)")
            
            if aurc_lift >= 0.15 and r200_diff >= 0.10:
                print("\nSUCCESS: Acceptance Criteria Met!")
            else:
                print("\nFAILURE: Acceptance Criteria NOT Met.")

if __name__ == "__main__":
    analyze()
