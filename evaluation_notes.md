# Evaluation Notes

## Memory Efficiency Baselines

The current implementation of the memory efficiency benchmark in `src/evaluation/efficiency_eval.py` measures the `bytes_per_fact` and plots accuracy vs. memory usage for the DBME model. However, it does not yet include the requested comparisons against baseline models such as:

- No-memory LM (vanilla GPT-2 small)
- Transformer KV Cache (standard sliding window)
- Compressive Transformer
- FAISS Retrieval Baseline (vector DB retrieval without consolidation)
- RETRO-lite baseline

These baselines are important for a comprehensive evaluation of the memory efficiency of the DBME model. They have been omitted for now to focus on the core implementation of the evaluation harness, but they should be added in a future iteration to provide a more complete picture of the model's performance.