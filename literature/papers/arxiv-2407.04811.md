---
id: arxiv-2407.04811
title: "Simplifying Deep Temporal Difference Learning"
authors: "Gallici M, Fellows M, Ellis B, Pou B, Masmitja I, Foerster JN, Martin M"
venue: "arXiv:2407.04811v2, 23 Oct 2024 (PDF header: under review at ICLR 2025)"
peer_reviewed: preprint
date_published: 2024
date_reviewed: 2026-09-23
pdf_url: https://arxiv.org/abs/2407.04811
provenance: 1
status: reviewed
relevance_tier: 4

object: theory
model_family: none
observability: n/a
endpoint: none
seeds: "10; 3 on ALE (p.30)"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "PQN: Q(lambda) on parallel environments with a LayerNorm Q-network, no replay buffer and no target network (Alg. 1 p.7, Alg. 2 p.30); RNN-compatible by BPTT over collected trajectories (p.7)"
verdict_line: "Adopted as the Phase 2c trainer for BOTH arms after PPO left the ODE-RNN open-loop (L312, L391); its stability theorem is a wide-network result, which our 16-unit ODE hidden state is not."
---

# Simplifying Deep Temporal Difference Learning (PQN)

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2407.04811.pdf`
> (v2, 39 pp.). Page numbers are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `theory` + algorithm (PQN) |
| S2 | Model family | `none` — games and control benchmarks |
| S3 | Observability | `n/a` |
| S4 | Endpoint | `none` |
| S5 | Seeds | 10 seeds; 3 on ALE (p.30) |

## What it does

Replaces DQN's replay buffer and target network with (i) many environments
sampled in parallel, online, and (ii) a LayerNorm-regularised Q-network
(Alg. 1, p.7). Targets are Peng's Q(λ), computed backward over a short
collected trajectory: `R_t = r_t + γ[λ R_{t+1} + (1−λ) max_a' Q(s',a')]`
(p.7; Alg. 2, p.30). Exploration is ε-greedy with linear decay (Alg. 1, p.7;
Table 4, p.36). Optimiser RAdam, reported as more robust to its ε, not better
(p.30).

## What transfers

- **No replay buffer ⇒ recurrent policies need no stored hidden states or
  burn-in**: "an implementation of PQN using RNNs only requires sampling
  trajectories for multiple time-steps and then back-propagating the gradient
  through time" (p.7). This is the property that makes it usable with our
  ODE-RNN where SAC/TD3 would need R2D2-style replay.
- **Recurrent hyperparameters** (Craftax RNN, Table 4, p.36): ε 1.0 → 0.005 over
  the first 10% of updates, 4 epochs × 4 minibatches, LR 3e-4 with linear
  decay, max-grad-norm 0.5, γ 0.99, λ 0.5.
- **λ matters**: on Atari-10 λ=0.65 beat λ=0 "by a significant margin"
  (Fig. 5c, p.10).
- **LayerNorm over BatchNorm** in general (Fig. 5a, p.10); for Craftax, input
  BatchNorm or LayerNorm on the rest of the network gave similar gains (p.10).
- **RNN evidence is thin**: one environment, PQN-RNN 16% vs PPO-RNN 15.3%
  on Craftax (p.9), with hyperparameters tuned for both (p.31).

## What does not

- **The stability theorem is a wide-network limit.** Theorem 2 (p.6) holds
  "for all k > k′", where k is the critic's width and nonlinear instability
  vanishes "as k → ∞" (p.6, Lemma 1 / Eq. 8, p.5). It also assumes ℓ2 on the
  final layer (Eq. 9, p.6). Our MLP Q-layer would be ~441 wide; the ODE-RNN's
  hidden state is 16. The guarantee plausibly covers one arm far better than
  the other — declare it, do not assume it.
- **ε-greedy is weak exploration**: PQN underperforms on hard-exploration ALE
  games, which the authors attribute to ε-greedy being "too simple" (p.9).
- **Minibatches of random transitions** (Alg. 2 line 9, p.30) are only valid for
  feedforward networks; for a recurrent Q-network the unit must be the whole
  trajectory, as p.7's BPTT description implies.
- Benchmarks are games with full or image observations; none is a
  burden-only POMDP.

## Contact with our design

PROJECT.md §8/§12: the architecture is swapped inside ONE algorithm with
everything else held fixed. PQN replaces PPO for both arms in Phase 2c, with
Table 4's recurrent settings adopted rather than tuned. Two deviations to
declare in the registration: LayerNorm (not BatchNorm) on the Q-layer, and no
last-action input (Table 4's `ADD_LAST_ACTION`), so the policy interface stays
identical across arms.
