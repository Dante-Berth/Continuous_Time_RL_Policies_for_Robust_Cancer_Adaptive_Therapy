"""Scripted dosing rules behind the policy interface -- reference points, not arms.

They run through the same closed-loop rollout as the trained policies, so a rule
and a network differ only in the policy object. Each emits one-hot logits and is
evaluated greedily. `obs` is log(burden / baseline) (policies/base.py).

  none     never treat
  MTD      full dose at every visit (standard of care)
  AT50     Zhang et al.'s trial rule: treat until burden < 50% of baseline, hold
           until it is back at baseline. Hysteresis, carried in the state.
  thr0.8   full dose whenever burden > 0.8 x baseline -- the switch Gallagher
           2024's RL learned (N* = 0.83 n0, p.1935) and our PQN MLP learned in
           W1 (RESEARCH_LOG L461)

PREREG W3c [2026-09-24] reports them descriptively beside every arm.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_therapy.policies.base import N_ACTIONS

RULES = ("none", "MTD", "AT50", "thr0.8")


class RulePolicy(eqx.Module):
    kind: str = eqx.field(static=True)

    def __init__(self, kind: str):
        if kind not in RULES:
            raise ValueError(f"unknown rule {kind!r}; expected one of {RULES}")
        self.kind = kind

    def initial_state(self) -> jnp.ndarray:
        return jnp.ones((1,))  # AT50 starts ON; the other rules ignore it

    def __call__(self, obs, state, dt, key=None):
        on = state[0]
        if self.kind == "none":
            a = jnp.array(0)
        elif self.kind == "MTD":
            a = jnp.array(N_ACTIONS - 1)
        elif self.kind == "thr0.8":
            a = jnp.where(obs > jnp.log(0.8), N_ACTIONS - 1, 0)
        else:  # AT50
            on = jnp.where(obs < jnp.log(0.5), 0.0, jnp.where(obs >= 0.0, 1.0, on))
            a = jnp.where(on > 0.5, N_ACTIONS - 1, 0)
        return jax.nn.one_hot(a, N_ACTIONS) * 10.0, jnp.array([on])

    def value(self, obs, state) -> jnp.ndarray:
        return jnp.array(0.0)


class ThresholdPolicy(eqx.Module):
    """Full dose when burden > `level` x baseline, else none: M1's policy class.

    `level` is an array leaf, not a static field, so one compiled rollout
    serves every level. ThresholdPolicy(0.8) doses exactly like RulePolicy("thr0.8").
    """

    level: jnp.ndarray

    def __init__(self, level: float):
        self.level = jnp.asarray(level, dtype=jnp.float32)

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((0,))

    def __call__(self, obs, state, dt, key=None):
        a = jnp.where(obs > jnp.log(self.level), N_ACTIONS - 1, 0)
        return jax.nn.one_hot(a, N_ACTIONS) * 10.0, state

    def value(self, obs, state) -> jnp.ndarray:
        return jnp.array(0.0)
