"""Shared types for the tumour ODE families.

A virtual patient is `(model_id, theta)`; a cohort is arrays thereof. Every
family exposes the same signature so `solve_cohort` can vmap over any of them:

    vector_field(t, y, args) -> dy/dt

where `args` is a `(theta, dose_fn)` pair. Families differ in the meaning and
length of `theta` and in the state dimension, never in the call shape.

PROJECT.md 7: the observable is TOTAL BURDEN ONLY. `observe()` is the only
function a policy may see, and it deliberately collapses the state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import jax.numpy as jnp

ModelId = Literal["lotka_volterra", "hahnfeldt", "kuznetsov"]


@dataclass(frozen=True)
class SolverConfig:
    """How a cohort is integrated.

    Defaults are FIXED-STEP on purpose. PROJECT.md 12: vmap over an
    adaptive-step solver does not give each patient its own step count -- under
    XLA the whole batch pays the stiffest patient's step count. Fixed-step for
    the bulk; the stiff bucket is solved separately with an implicit solver.
    """

    t0: float = 0.0
    t1: float = 365.0
    dt0: float = 0.1
    n_save: int = 366
    stiff: bool = False  # True -> Kvaerno5 for the stiff bucket only
    max_steps: int = 100_000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SolverConfig:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass(frozen=True)
class PKConfig:
    """Minimal single-compartment PK: dose -> effective concentration.

    Deliberately KNOWN and white-box (PROJECT.md 7). The research question is
    about the policy, not about learning pharmacokinetics.
    """

    clearance: float = 0.5  # 1/day; exponential decay of effective concentration
    bioavailability: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def observe(state: jnp.ndarray, model_id: ModelId) -> jnp.ndarray:
    """Collapse a full state to the scalar burden a policy is allowed to see.

    This is the POMDP boundary. Anything that reads `state` directly instead of
    calling this is a leak, and makes every robustness claim meaningless.
    """
    if model_id == "lotka_volterra":
        return state[..., 0] + state[..., 1]  # S + R
    if model_id == "hahnfeldt":
        return state[..., 0]  # V
    if model_id == "kuznetsov":
        return state[..., 1]  # T (effector cells are not measurable)
    raise ValueError(f"unknown model_id: {model_id}")
