# ACTIVE RUNS — what is running where

Operational state only (host, session, how to check, how to collect). Results go
to [RESEARCH_LOG.md](adaptive_therapy/RESEARCH_LOG.md) via `/logentry`, never
here. Remove a row once its log entry exists. Not part of the precedence ladder.

| Started | Host | Run | Registration | tmux session | Output (on host) |
|---|---|---|---|---|---|
| 2026-10-21 14:28 | sureli1 | V13 stage 1 (gate): r2d2_mlp + r2d2_ref, 80 cells | [PREREG V13](adaptive_therapy/PREREG.md) | `v13_gate` | `research/outputs/v13_gate.json`, `v13_shards/` (progress: `tmux capture-pane -pt v13_gate`) |

Check:    `ssh sureli1 'tmux ls; ls ~/Continuous_Time_RL_Policies_for_Robust_Cancer_Adaptive_Therapy/research/outputs/v13_shards | wc -l'`
Collect:  `rsync -a sureli1:Continuous_Time_RL_Policies_for_Robust_Cancer_Adaptive_Therapy/research/outputs/v13_* research/outputs/`

Stage 2 (`--stage compare`, adds LSTM and ODE-RNN) runs ONLY if the gate reading is `passed` in `v13_gate.json`.
