Nature of contribution: 
Not new theory; engineering-first paper. 
Main idea is state-augmented RL: enrich the RL state with extra predictive signals.


How they do it: 
Combine price-based classifier (LSTM labels) + news embeddings (GloVe+HAN) with a standard RL policy (DPG/PPO/PG). 
It’s a plug-and-play stack, not a new math formula.


Why it works: 
Even noisy (~60%) but dense side-signals shift action priors → better PV/SR; density > sparsity.


What to copy: 
Treat side-signals (order-flow imbalance, vol regime, microstructure stats) as a separate channel δ; 
Keep strict no-leak timing; 
Run regime/stress tests.


Caveats: 
Key Issue: Reproducibility (no code) → not clear of dataset splitting strictly or not ? (train/valid/test/unseen) 
    Price-trend classifier might be chosen to good on test-set
    And become future price-trend leaking to RL-Agent 
Simplified frictions, offline-RL bias; 
Guard with embargoed walk-forward + cost/latency sensitivity.



