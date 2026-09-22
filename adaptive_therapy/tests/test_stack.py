"""The stack resolves and JAX transforms compose over the solver.

PROJECT.md 12 locks equinox+diffrax+optax specifically because a network is a
pytree that drops straight into a diffrax vector field. This asserts that
property holds in the installed versions, so a dependency bump that breaks it
fails here rather than three phases later inside an adjoint.
"""

import diffrax
import jax
import jax.numpy as jnp


def test_vmap_composes_over_diffrax_solve():
    """A batched solve returns one trajectory per parameter draw."""

    def solve(rate):
        term = diffrax.ODETerm(lambda t, y, args: -args * y)
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Dopri5(),
            t0=0.0,
            t1=1.0,
            dt0=0.1,
            y0=jnp.array(1.0),
            args=rate,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 1.0, 5)),
        )
        return sol.ys

    rates = jnp.array([0.5, 1.0, 2.0])
    out = jax.vmap(solve)(rates)

    assert out.shape == (3, 5)
    assert jnp.all(jnp.isfinite(out))
    # Decay: a larger rate must leave less mass at t1.
    assert out[0, -1] > out[1, -1] > out[2, -1]
