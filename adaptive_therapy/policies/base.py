"""Policy interface. The ONE thing that varies in the Phase 2/3 ablation.

Every architecture exposes the same call shape:

    policy(obs, state, dt, key) -> (logits, new_state)

`obs` is the scalar burden observation (PROJECT.md 7 -- never subpopulations).
`state` is whatever recurrent state the architecture carries; `dt` is the time
since the previous observation, in days.

`dt` is in the signature for EVERY architecture, including the memoryless MLP
that ignores it. That is deliberate: it keeps the RL loop identical across arms
so the ablation varies the architecture and nothing else (PROJECT.md 8), and it
is what makes the Tier-4 irregular-sampling experiment a drop-in rather than a
rewrite.
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

N_ACTIONS = 3  # dose in {0, 1/2, 1} -- boluses, not infusion (PROJECT.md 7)
DOSE_LEVELS = jnp.array([0.0, 0.5, 1.0])


class Policy(Protocol):
    """Structural type every actor must satisfy."""

    def initial_state(self) -> jnp.ndarray: ...

    def __call__(
        self, obs: jnp.ndarray, state: jnp.ndarray, dt: float, key=None
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...


def normalise_obs(burden: jnp.ndarray, baseline: jnp.ndarray) -> jnp.ndarray:
    """Burden -> a scale-free observation the policy can generalise across.

    log-ratio to the patient's own baseline. Raw burden spans decades across
    virtual patients, so a network fed raw values would spend its capacity on
    scale rather than on dynamics -- and, worse, would fail on a Tier-1 cohort
    whose K is shifted, for reasons that have nothing to do with scheduling.
    """
    return jnp.log(jnp.maximum(burden, 1e-30) / jnp.maximum(baseline, 1e-30))
