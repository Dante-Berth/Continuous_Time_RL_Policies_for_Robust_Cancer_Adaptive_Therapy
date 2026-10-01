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

from adaptive_therapy.eval.metrics import DEFAULT_CONFIRM_DAYS


@dataclass(frozen=True)
class RewardConfig:
    # Paid at EVERY visit, progressed or not, and episodes never terminate, so
    # over an episode it is a constant: the progression signal is the burden
    # term alone. 2026-09-29 (RESEARCH_LOG L739): this comment used to say "per
    # day alive and not progressed", and W3c's prereg called U "the trained
    # objective" on that basis. Code unchanged; every logged run used it.
    survival_bonus: float = 1.0
    burden_penalty: float = 1.0  # per unit relative burden above threshold
    exposure_penalty: float = 0.1  # per unit dose; report at >= 2 values
    progression_threshold: float = 1.2  # relative to the running nadir
    # "always": the survival term is paid every visit (every run up to T1).
    # "progression": it stops once progression is CONFIRMED online, so the
    # episode's survival total is the TTP plus the confirmation window -- the
    # spec's "+TTP" (PROJECT.md 7). 2026-10-02, PREREG V2: T1 (RESEARCH_LOG
    # L1042) found the reference ties the MLP on the old reward and still loses
    # on U, because no term of that reward is TTP (L739).
    survival_until: str = "always"
    confirm_days: float = DEFAULT_CONFIRM_DAYS  # the TTP rule's window

    def to_dict(self) -> dict:
        return asdict(self)


def progression_update(
    run: jnp.ndarray,
    progressed: jnp.ndarray,
    burden: jnp.ndarray,
    nadir: jnp.ndarray,
    dt: float,
    cfg: RewardConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One visit of the online TTP rule -> (run', progressed').

    `run` counts consecutive visits above `progression_threshold` x running
    nadir; progression is confirmed once that run spans `confirm_days`. On a
    regular grid this flips exactly `confirm_days` after the crossing that
    eval/metrics.ttp_running_nadir reports, except in the last `confirm_days`
    of the horizon, where the offline rule counts an unfalsified window as
    confirmed and an online rule cannot.
    """
    above = burden > nadir * cfg.progression_threshold
    run2 = jnp.where(above, run + 1, 0)
    return run2, progressed | ((run2 - 1) * dt >= cfg.confirm_days)


def step_reward(
    burden: jnp.ndarray,
    nadir: jnp.ndarray,
    dose: jnp.ndarray,
    dt: float,
    cfg: RewardConfig,
    progressed: jnp.ndarray | bool = False,
) -> jnp.ndarray:
    """Reward for one decision interval.

    Burden is penalised relative to the RUNNING NADIR, matching the TTP rule in
    eval/metrics.py. If the reward used a different progression notion than the
    endpoint, the agent would be optimising something the paper does not
    report -- the most insidious version of optimising the wrong thing.
    """
    alive = cfg.survival_bonus * dt
    if cfg.survival_until == "progression":
        alive = jnp.where(progressed, 0.0, alive)
    elif cfg.survival_until != "always":
        raise ValueError(f"unknown survival_until {cfg.survival_until!r}")

    rel = burden / jnp.maximum(nadir, 1e-10)
    excess = jnp.maximum(rel - cfg.progression_threshold, 0.0)
    burden_cost = cfg.burden_penalty * excess * dt

    exposure_cost = cfg.exposure_penalty * dose * dt

    return alive - burden_cost - exposure_cost
