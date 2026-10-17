"""Memory dropout for a recurrent policy (PREREG V10).

V9 (RESEARCH_LOG L1477) removed the positional MLP's whole deficit by hiding its
clock in half of the training episodes. This is the same treatment for the
recurrent arms: in the episodes whose first observation is positive (about
half), the wrapped policy's recurrent state is reset to its initial value at
every visit, so it sees only the current observation and must act as a
memoryless net. In the other half it is the unchanged LSTM / ODE-RNN. The flag
comes from the sign of the first observation (pure noise at visit 0), carried in
the state, so rollout and replay agree without plumbing a key -- the same rule
as MLPPolicy(clock_dropout=True). No parameters are added.

`forced("hide")` / `forced("show")` pin the flag at test; the primary test is
"show" (memory always available).
"""

from __future__ import annotations

import copy

import equinox as eqx
import jax.numpy as jnp


class StateDropout(eqx.Module):
    inner: eqx.Module
    force: str = eqx.field(static=True, default="")

    def initial_state(self) -> jnp.ndarray:
        return jnp.concatenate([self.inner.initial_state(), jnp.zeros(2)])

    def forced(self, mode: str) -> "StateDropout":
        new = copy.copy(self)
        object.__setattr__(new, "force", mode)
        return new

    def _split(self, state):
        return state[:-2], state[-2], state[-1]

    def _hidden(self, flag):
        return {"hide": 1.0, "show": 0.0}.get(self.force, flag)

    def __call__(self, obs, state, dt, key=None):
        h, flag, started = self._split(state)
        flag = jnp.where(started > 0.5, flag, (jnp.atleast_1d(obs)[0] > 0).astype(state.dtype))
        h_in = jnp.where(self._hidden(flag) > 0.5, self.inner.initial_state(), h)
        logits, h_out = self.inner(obs, h_in, dt)
        return logits, jnp.concatenate([h_out, flag[None], jnp.ones(1, state.dtype)])

    def value(self, obs, state) -> jnp.ndarray:
        h, _, _ = self._split(state)
        return self.inner.value(obs, h)
