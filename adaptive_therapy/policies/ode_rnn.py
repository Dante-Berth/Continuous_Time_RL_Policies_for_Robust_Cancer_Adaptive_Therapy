"""ODE-RNN actor-critic -- the continuous-time policy under test.

Rubanova et al. (NeurIPS 2019): the hidden state obeys an ODE *between*
observations and jumps *at* them.

    h(t_k^-) = ODESolve(f_theta, h(t_{k-1}), dt)      # flow across the gap
    h(t_k)   = GRUCell(obs_k, h(t_k^-))               # jump at the observation

Why this is the right continuous-time baseline and not a bespoke design
(PROJECT.md 8): it is the published encoder for irregularly-sampled series, it
is the Q4 baseline a pure-liquid policy would have to beat, and reinventing it
would make the comparison unreadable. Reference implementation is PyTorch; this
is a port to equinox/diffrax.

The architectural claim being tested is narrow and falsifiable: integrating the
inter-visit interval explicitly should beat treating it as one opaque step. The
MLP arm differs in exactly that respect and in no other.
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_therapy.policies.base import N_ACTIONS


class ODERNNPolicy(eqx.Module):
    vf: list  # the latent vector field f_theta
    cell: eqx.nn.GRUCell
    head: eqx.nn.Linear
    critic: eqx.nn.Linear
    hidden: int = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(self, key, width: int = 64, hidden: int = 16):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.hidden = hidden
        # Fixed-step flow with a bounded step count. Phase 1 measured that an
        # adaptive controller on this simulator thrashes and can fail outright;
        # inside a training loop the same failure would appear as a mysterious
        # NaN thousands of updates in.
        self.max_steps = 64
        self.vf = [
            eqx.nn.Linear(hidden, width, key=k1),
            eqx.nn.Linear(width, hidden, key=k2),
        ]
        self.cell = eqx.nn.GRUCell(1, hidden, key=k3)
        self.head = eqx.nn.Linear(hidden, N_ACTIONS, key=k4)
        self.critic = eqx.nn.Linear(hidden, 1, key=k5)

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((self.hidden,))

    def _field(self, t, h, args):
        # tanh on the output bounds the drift: an unbounded latent field over a
        # 7-day gap is how a continuous-time policy diverges during training.
        return jnp.tanh(self.vf[1](jax.nn.tanh(self.vf[0](h))))

    def flow(self, h: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Evolve the hidden state across an observation gap of `dt` days."""
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._field),
            diffrax.Tsit5(),
            t0=0.0,
            t1=dt,
            dt0=dt / 8.0,
            y0=h,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=self.max_steps,
        )
        return sol.ys[-1]

    def __call__(self, obs, state, dt, key=None):
        h = jax.lax.cond(dt > 0.0, lambda: self.flow(state, dt), lambda: state)
        h = self.cell(jnp.atleast_1d(obs), h)
        return self.head(h), h

    def value(self, obs, state) -> jnp.ndarray:
        return self.critic(state)[0]
