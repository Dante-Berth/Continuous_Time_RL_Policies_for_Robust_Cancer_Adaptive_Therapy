"""Memoryless MLP actor-critic -- the discrete baseline.

This is the null hypothesis for PROJECT.md Q2, and it is deliberately a FAIR
baseline rather than a weak one: same width, same learning rate, same observation
preprocessing and same RL algorithm as the ODE arm. PROJECT.md 13 names
"beating only weak baselines" as a reviewer reflex, and the cheapest way to earn
that criticism is to hobble the MLP.

It ignores `dt` because it has no state to integrate. That is the architectural
difference being tested, not an oversight.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_therapy.policies.base import N_ACTIONS


class MLPPolicy(eqx.Module):
    layers: list
    critic: list
    norm: eqx.nn.LayerNorm | None
    width: int = eqx.field(static=True)

    def __init__(self, key, width: int = 384, layer_norm: bool = False):
        # 384, not 64: it is what makes this arm parameter-matched to the
        # ODE-RNN (3076 each, ratio 1.00). See the PREREG amendment
        # [2026-09-22]. A 64-wide MLP has 516 params against the ODE arm's
        # 3076, and a win for the bigger network would be uninterpretable.
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.width = width
        self.layers = [
            eqx.nn.Linear(1, width, key=k1),
            eqx.nn.Linear(width, N_ACTIONS, key=k2),
        ]
        # Optional LayerNorm on the hidden layer: the Q-network regulariser PQN's
        # stability argument relies on (arxiv-2407.04811 p.6). Off for PPO.
        self.norm = eqx.nn.LayerNorm(width) if layer_norm else None
        self.critic = [
            eqx.nn.Linear(1, width, key=k3),
            eqx.nn.Linear(width, 1, key=k4),
        ]

    def initial_state(self) -> jnp.ndarray:
        """No recurrent state. Returned as a length-0 array so the RL loop can
        carry a state for every architecture without special-casing."""
        return jnp.zeros((0,))

    def __call__(self, obs, state, dt, key=None):
        x = jnp.atleast_1d(obs)
        z = self.layers[0](x)
        h = jax.nn.tanh(self.norm(z) if self.norm is not None else z)
        return self.layers[1](h), state

    def value(self, obs, state) -> jnp.ndarray:
        x = jnp.atleast_1d(obs)
        h = jax.nn.tanh(self.critic[0](x))
        return self.critic[1](h)[0]
