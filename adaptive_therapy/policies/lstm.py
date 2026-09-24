"""LSTM Q-network -- the discrete-time recurrent arm (PROJECT.md 8, arm b).

It separates the two things Phase 2 confounded (RESEARCH_LOG L424): against the
MLP it adds MEMORY, against the ODE-RNN it lacks CONTINUOUS TIME. It ignores
`dt` exactly as the MLP does: one visit is one opaque step. The LSTM + dt input
variant is W4's baseline, not this arm.

PREREG W3 [2026-09-24]: hidden 26 with the same LayerNorm readout as the other
arms puts its Q-path within 5% of their 3090 parameters.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_therapy.policies.base import N_ACTIONS


class LSTMPolicy(eqx.Module):
    cell: eqx.nn.LSTMCell
    head: eqx.nn.Linear
    critic: eqx.nn.Linear
    norm: eqx.nn.LayerNorm | None
    hidden: int = eqx.field(static=True)

    def __init__(self, key, hidden: int = 26, layer_norm: bool = False):
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden = hidden
        self.cell = eqx.nn.LSTMCell(1, hidden, key=k1)
        self.head = eqx.nn.Linear(hidden, N_ACTIONS, key=k2)
        self.norm = eqx.nn.LayerNorm(hidden) if layer_norm else None
        self.critic = eqx.nn.Linear(hidden, 1, key=k3)

    def initial_state(self) -> jnp.ndarray:
        """(h, c) concatenated, so the RL loop carries one array for every arm."""
        return jnp.zeros((2 * self.hidden,))

    def __call__(self, obs, state, dt, key=None):
        h, c = self.cell(jnp.atleast_1d(obs), (state[: self.hidden], state[self.hidden :]))
        return self.head(self.norm(h) if self.norm is not None else h), jnp.concatenate([h, c])

    def value(self, obs, state) -> jnp.ndarray:
        return self.critic(state[: self.hidden])[0]
