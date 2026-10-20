"""de Pillis 2006 tumour-immune-chemotherapy model: invariants, not numerics."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from adaptive_therapy.envs import depillis as dp  # noqa: E402

MOUSE_Y0 = jnp.array([1e6, 5e4, 100.0, 1.1e7, 0.0, 0.0])


def _zeros(n):
    return jnp.zeros(n)


def test_every_preset_defines_every_parameter_and_is_positive():
    for name, params in dp.PRESETS.items():
        theta = dp.theta_from(params)
        assert theta.shape == (dp.N_THETA,), name
        assert bool(jnp.all(theta > 0)), name


def test_chemo_toxicity_is_built_in_immune_kill_is_below_tumour_kill():
    # The reason this family is in the program (PREREG W13): full dosing is not free.
    for params in dp.PRESETS.values():
        assert 0 < params["K_N"] <= params["K_T"]
        assert params["K_L"] == params["K_N"] == params["K_C"]


def test_drug_decays_exponentially_without_input():
    theta = dp.theta_from(dp.MOUSE)
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    ys = dp.simulate_days(theta, y0, _zeros(5), _zeros(5), _zeros(5))
    expected = 2.0 * np.exp(-dp.MOUSE["gamma"] * np.arange(6))
    np.testing.assert_allclose(np.asarray(ys[:, 4]), expected, rtol=1e-5)


def test_tumour_free_immune_rest_state_is_an_equilibrium_without_treatment():
    theta = dp.theta_from(dp.PATIENT9)
    c_star = dp.PATIENT9["alpha"] / dp.PATIENT9["beta"]
    n_star = dp.PATIENT9["e"] * c_star / dp.PATIENT9["f"]
    y = jnp.array([0.0, n_star, 0.0, c_star, 0.0, 0.0])
    dy = dp.vector_field(0.0, y, (theta, (0.0, 0.0, 0.0)))
    assert float(jnp.abs(dy[1])) < 1e-6 * n_star
    assert float(jnp.abs(dy[3])) < 1e-6 * c_star
    assert float(dy[0]) == 0.0


def test_chemotherapy_depletes_circulating_lymphocytes():
    # Toxicity: with no tumour, repeated dosing must pull C below its untreated rest value.
    theta = dp.theta_from(dp.PATIENT9)
    c_star = dp.PATIENT9["alpha"] / dp.PATIENT9["beta"]
    y0 = jnp.array([0.0, 1.0, 0.0, c_star, 0.0, 0.0])
    ys = dp.simulate_days(theta, y0, jnp.full(9, 5.0), _zeros(9), _zeros(9))
    assert float(ys[-1, 3]) < 0.99 * c_star


def test_til_pulse_adds_its_cells_to_the_cd8_pool_when_nothing_suppresses_them():
    # NK suppression (u N L^2) is switched off: with it on, the NK cells that the
    # lymphocyte source creates within the day remove most of a 8e8 bolus.
    theta = dp.theta_from(dict(dp.MOUSE, u=0.0))
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ys = dp.simulate_days(theta, y0, _zeros(1), jnp.array([8e8]), _zeros(1))
    # one day at 8e8/day with death rate m = 0.02/day: 8e8 (1 - e^-m) / m
    expected = 8e8 * (1 - np.exp(-dp.MOUSE["m"])) / dp.MOUSE["m"]
    np.testing.assert_allclose(float(ys[-1, 2]), expected, rtol=1e-4)


def test_untreated_rollout_is_finite_and_nonnegative_for_all_presets():
    for params in dp.PRESETS.values():
        theta = dp.theta_from(params)
        ys = dp.simulate_days(theta, MOUSE_Y0, _zeros(60), _zeros(60), _zeros(60))
        assert bool(jnp.isfinite(ys).all())
        assert float(ys.min()) > -1e-6


def test_fixed_step_agrees_with_the_adaptive_solver_on_a_chemo_course():
    theta = dp.theta_from(dp.PATIENT9)
    y0 = jnp.array([2e7, 1e3, 10.0, 6e8, 0.0, 0.0])
    v_m = jnp.zeros(40).at[::5].set(5.0)
    a = dp.simulate_days(theta, y0, v_m, _zeros(40), _zeros(40))
    b = dp.simulate_days(theta, y0, v_m, _zeros(40), _zeros(40), fixed_dt=0.05)
    np.testing.assert_allclose(
        np.log10(np.asarray(b[:, 0]) + 1.0), np.log10(np.asarray(a[:, 0]) + 1.0), atol=2e-3
    )
    np.testing.assert_allclose(np.asarray(b[:, 3]), np.asarray(a[:, 3]), rtol=1e-3)


def test_sampled_patients_stay_in_the_registered_box_and_cycle_the_three_sets():
    theta, y0 = dp.sample_patients(jax.random.PRNGKey(0), 90)
    assert theta.shape == (90, dp.N_THETA) and y0.shape == (90, 6)
    for i, (lo, hi) in enumerate(dp.IC_BOX.values()):
        assert float(y0[:, i].min()) >= lo * 0.999 and float(y0[:, i].max()) <= hi * 1.001
    assert bool(jnp.all(y0[:, 4:] == 0.0))
    assert len({tuple(map(float, r)) for r in theta}) == 3
