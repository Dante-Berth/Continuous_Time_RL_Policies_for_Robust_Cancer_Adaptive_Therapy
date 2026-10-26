# ACTIVE RUNS — what is running where

Operational state only (host, session, how to check, how to collect). Results go
to [RESEARCH_LOG.md](adaptive_therapy/RESEARCH_LOG.md) via `/logentry`, never
here. Remove a row once its log entry exists. Not part of the precedence ladder.

| Started | Host | Run | Registration | tmux session | Output (on host) |
|---|---|---|---|---|---|

Check:    `ssh sureli1 'tmux ls'`, then `ls` the run's shard directory under `research/outputs/` on the host
Collect:  `rsync -a sureli1:Continuous_Time_RL_Policies_for_Robust_Cancer_Adaptive_Therapy/research/outputs/<run>_* research/outputs/`

V13 stage 1 finished 2026-10-21 (gate failed, [L1651](adaptive_therapy/RESEARCH_LOG.md#L1651)); stage 2 is not to be run.

V14 stage 1 finished 2026-10-22 (gate failed, [L1668](adaptive_therapy/RESEARCH_LOG.md#L1668)); stage 2 is not to be run.

V15 stage 1 (burden-integral re-score of saved shards) ran locally and finished 2026-10-22 (failed, [L1685](adaptive_therapy/RESEARCH_LOG.md#L1685)); stage 2 is not to be run.
