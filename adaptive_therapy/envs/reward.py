"""Reward: survival time, penalised for burden and for cumulative exposure.

PROJECT.md 7:
  +TTP; -penalty for burden above baseline (e.g. +20%); -cumulative-exposure
  penalty. "Without it the agent cheats by dosing constantly."

The exposure weight is NOT a tuning knob to be optimised for the best-looking
result. PROJECT.md 7 requires reporting at >= 2 exposure weights, because the
TTP/exposure trade-off is the finding, not a hyperparameter to select over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class RewardConfig:
    survival_bonus: float = 1.0  # per day alive and not progressed
    burden_penalty: float = 1.0  # per unit relative burden above threshold
    exposure_penalty: float = 0.1  # per unit dose; report at >= 2 values
    progression_threshold: float = 1.2  # relative to the running nadir

    def to_dict(self) -> dict:
        return asdict(self)


def step_reward(
    burden: jnp.ndarray,
    nadir: jnp.ndarray,
    dose: jnp.ndarray,
    dt: float,
    cfg: RewardConfig,
) -> jnp.ndarray:
    """Reward for one decision interval.

    Burden is penalised relative to the RUNNING NADIR, matching the TTP rule in
    eval/metrics.py. If the reward used a different progression notion than the
    endpoint, the agent would be optimising something the paper does not
    report -- the most insidious version of optimising the wrong thing.
    """
    alive = cfg.survival_bonus * dt

    rel = burden / jnp.maximum(nadir, 1e-10)
    excess = jnp.maximum(rel - cfg.progression_threshold, 0.0)
    burden_cost = cfg.burden_penalty * excess * dt

    exposure_cost = cfg.exposure_penalty * dose * dt

    return alive - burden_cost - exposure_cost
