"""Memoryless MLP actor-critic -- the discrete baseline.

This is the null hypothesis for PROJECT.md Q2, and it is deliberately a FAIR
baseline rather than a weak one: same width, same learning rate, same observation
preprocessing and same RL algorithm as the ODE arm. PROJECT.md 13 names
"beating only weak baselines" as a reviewer reflex, and the cheapest way to earn
that criticism is to hobble the MLP.

It ignores `dt` because it has no state to integrate. That is the architectural
difference being tested, not an oversight.

`clock=True` is the positional MLP (PREREG V7): its only state is the elapsed
time, summed from `dt`, fed as a second input. It sees WHEN, never WHAT came
before. POPGym (arxiv-2303.01859 p.9) found this baseline often beats the
plain MLP and calls it standard for memory comparisons.
"""

from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_therapy.policies.base import N_ACTIONS

CLOCK_SCALE = 357.0  # days per unit of the clock input: 1.0 at the 52-visit horizon


class MLPPolicy(eqx.Module):
    layers: list
    critic: list
    norm: eqx.nn.LayerNorm | None
    width: int = eqx.field(static=True)
    clock: bool = eqx.field(static=True)
    clock_dropout: bool = eqx.field(static=True)
    clock_force: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        width: int = 384,
        layer_norm: bool = False,
        in_dim: int = 1,
        clock: bool = False,
        clock_dropout: bool = False,
    ):
        # 384, not 64: it is what makes this arm parameter-matched to the
        # ODE-RNN (3076 each, ratio 1.00). See the PREREG amendment
        # [2026-09-22]. A 64-wide MLP has 516 params against the ODE arm's
        # 3076, and a win for the bigger network would be uninterpretable.
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.width = width
        # clock_dropout (PREREG V9): in half of the episodes the clock input is
        # replaced by its mid-episode value. Which half is decided by the sign of
        # the first observation (pure noise at visit 0), carried in the state, so
        # rollout and replay agree without plumbing a key. `clock_force` pins it
        # at test: "hide" (clock never shown) or "show" (always shown).
        self.clock = clock or clock_dropout
        self.clock_dropout = clock_dropout
        self.clock_force = ""
        in_dim += int(self.clock)
        # in_dim > 1 only for PREREG S2's full-information reference, which
        # also sees the hidden state (rollout.oracle_features); every arm is 1.
        self.layers = [
            eqx.nn.Linear(in_dim, width, key=k1),
            eqx.nn.Linear(width, N_ACTIONS, key=k2),
        ]
        # Optional LayerNorm on the hidden layer: the Q-network regulariser PQN's
        # stability argument relies on (arxiv-2407.04811 p.6). Off for PPO.
        self.norm = eqx.nn.LayerNorm(width) if layer_norm else None
        self.critic = [
            eqx.nn.Linear(in_dim, width, key=k3),
            eqx.nn.Linear(width, 1, key=k4),
        ]

    def initial_state(self) -> jnp.ndarray:
        """No recurrent state. Returned as a length-0 array so the RL loop can
        carry a state for every architecture without special-casing."""
        return jnp.zeros((3,) if self.clock_dropout else (1,) if self.clock else (0,))

    def forced(self, mode: str) -> "MLPPolicy":
        """A copy with the clock pinned at test: "hide", "show", or "" (the dropout)."""
        new = copy.copy(self)
        object.__setattr__(new, "clock_force", mode)
        return new

    def _shown(self, state):
        """The clock value the network sees: elapsed time, or mid-episode if hidden."""
        if not self.clock_dropout:
            return state[:1]
        hidden = {"hide": 1.0, "show": 0.0}.get(self.clock_force, state[1])
        return jnp.where(hidden > 0.5, CLOCK_SCALE / 2, state[:1])

    def _input(self, obs, t):
        x = jnp.atleast_1d(obs)
        return jnp.concatenate([x, t / CLOCK_SCALE]) if self.clock else x

    def __call__(self, obs, state, dt, key=None):
        if self.clock_dropout:
            flag = jnp.where(state[2] > 0.5, state[1], (jnp.atleast_1d(obs)[0] > 0).astype(state.dtype))
            state = jnp.stack([state[0] + jnp.asarray(dt, state.dtype), flag, jnp.ones_like(flag)])
        elif self.clock:
            state = state + jnp.asarray(dt, state.dtype)
        z = self.layers[0](self._input(obs, self._shown(state) if self.clock_dropout else state))
        h = jax.nn.tanh(self.norm(z) if self.norm is not None else z)
        return self.layers[1](h), state

    def value(self, obs, state) -> jnp.ndarray:
        t = self._shown(state) if self.clock_dropout else state
        h = jax.nn.tanh(self.critic[0](self._input(obs, t)))
        return self.critic[1](h)[0]


class ScrambledClock(eqx.Module):
    """A trained positional MLP whose clock input is replaced at test time (PREREG V8).

    The true elapsed time is carried as state, so the wrapper rolls out like
    the arm it wraps; only the value the network sees changes:
    `frozen` shows mid-episode at every visit (the clock carries no
    information), `reversed` shows the time remaining, `intact` is unchanged.
    """

    inner: MLPPolicy
    mode: str = eqx.field(static=True)

    def initial_state(self) -> jnp.ndarray:
        return self.inner.initial_state()

    def _seen(self, elapsed):
        if self.mode == "frozen":
            return jnp.full_like(elapsed, CLOCK_SCALE / 2)
        if self.mode == "reversed":
            return CLOCK_SCALE - elapsed
        return elapsed

    def __call__(self, obs, state, dt, key=None):
        elapsed = state + jnp.asarray(dt, state.dtype)
        q, _ = self.inner(obs, self._seen(elapsed) - jnp.asarray(dt, state.dtype), dt)
        return q, elapsed

    def value(self, obs, state) -> jnp.ndarray:
        return self.inner.value(obs, self._seen(state))
