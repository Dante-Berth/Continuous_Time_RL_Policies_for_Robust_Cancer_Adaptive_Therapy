# ACTIVE RUNS — what is running where

Operational state only (host, session, how to check, how to collect). Results go
to [RESEARCH_LOG.md](adaptive_therapy/RESEARCH_LOG.md) via `/logentry`, never
here. Remove a row once its log entry exists. Not part of the precedence ladder.

| Started | Host | Run | Registration | tmux session | Output (on host) |
|---|---|---|---|---|---|
| (filled when launched) | | | | | |

Check:    `ssh sureli1 'tmux ls; ls ~/Continuous_Time_RL_Policies_for_Robust_Cancer_Adaptive_Therapy/research/outputs/v13_shards | wc -l'`
Collect:  `rsync -a sureli1:Continuous_Time_RL_Policies_for_Robust_Cancer_Adaptive_Therapy/research/outputs/v13_* research/outputs/`
