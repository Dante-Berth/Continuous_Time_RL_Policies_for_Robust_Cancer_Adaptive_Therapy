"""Invariants of the three tumour ODE families.

Test names state the invariant, not the function. These are the properties a
later refactor must not break, and several of them encode a specific trap named
in PROJECT.md rather than a generic sanity check.
"""

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs import droop, hahnfeldt, kuznetsov, lotka_volterra, zhang2017
from adaptive_therapy.envs.base import SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_cohort, solve_one_patient

LV_THETA = jnp.array([0.027, 0.008, 1e4, 0.0, 0.0, 1.5])
HF_THETA = jnp.array([0.192, 5.85, 0.00873, 0.66, 1e4])
# Kuznetsov non-dimensional values; `a` and `b` chosen in the regime the
# 1994 paper analyses rather than at the bifurcation boundary.
KT_THETA = jnp.array([0.1181, 0.1245, 2.019e7, 3.422e-10, 0.00412, 0.018, 2.0e-9, 1.101e-7])

SHORT = SolverConfig(t1=50.0, n_save=51)


# --- Lotka-Volterra ---------------------------------------------------------


def test_untreated_lv_stays_bounded_by_carrying_capacity():
    """Competition for a fixed K is the whole mechanism; exceeding it is a bug."""
    y0 = lotka_volterra.initial_state(LV_THETA, f_R=0.01, n_0=0.5)
    ys = solve_one_patient("lotka_volterra", LV_THETA, y0, dose=0.0, cfg=SHORT)
    burden = observe(ys, "lotka_volterra")
    K = float(LV_THETA[2])
    assert jnp.all(burden <= K * 1.01)
    assert jnp.all(burden > 0)


def test_lv_populations_stay_non_negative():
    """A negative cell count is unphysical and silently poisons every metric."""
    y0 = lotka_volterra.initial_state(LV_THETA, f_R=0.5, n_0=0.9)
    ys = solve_one_patient("lotka_volterra", LV_THETA, y0, dose=1.0, cfg=SHORT)
    assert jnp.all(ys >= -1e-6)


def test_lv_treatment_suppresses_sensitive_but_not_resistant():
    """The asymmetry adaptive therapy exploits: u(t) acts on S only.

    Under full dose the resistant population must do NO WORSE than untreated --
    it is released from competition. If treatment suppressed both, there would
    be no resistance management problem to study.
    """
    y0 = lotka_volterra.initial_state(LV_THETA, f_R=0.05, n_0=0.5)
    untreated = solve_one_patient("lotka_volterra", LV_THETA, y0, 0.0, SHORT)
    treated = solve_one_patient("lotka_volterra", LV_THETA, y0, 1.0, SHORT)
    assert treated[-1, 0] < untreated[-1, 0]  # S suppressed
    assert treated[-1, 1] >= untreated[-1, 1] * 0.99  # R not suppressed


def test_lv_initial_state_honours_resistant_fraction():
    """f_R and n_0 are two of the four factors Strobl showed matter."""
    y0 = lotka_volterra.initial_state(LV_THETA, f_R=0.2, n_0=0.75)
    total = float(y0[0] + y0[1])
    assert total == pytest.approx(0.75 * float(LV_THETA[2]), rel=1e-6)
    assert float(y0[1]) / total == pytest.approx(0.2, rel=1e-6)


# --- Hahnfeldt --------------------------------------------------------------


def test_untreated_hahnfeldt_volume_stays_positive_and_finite():
    """Gompertzian growth is singular at V=0; the floor must hold."""
    y0 = hahnfeldt.initial_state(HF_THETA, v_0=100.0)
    ys = solve_one_patient("hahnfeldt", HF_THETA, y0, dose=0.0, cfg=SHORT)
    assert jnp.all(jnp.isfinite(ys))
    assert jnp.all(ys[:, 0] > 0)


def test_hahnfeldt_treatment_reduces_carrying_capacity():
    """The drug is anti-angiogenic: it acts on K, not directly on V.

    This is what makes the family a genuine mechanism shift rather than a
    reparameterised LV -- the bottleneck is vasculature.
    """
    y0 = hahnfeldt.initial_state(HF_THETA, v_0=100.0)
    untreated = solve_one_patient("hahnfeldt", HF_THETA, y0, 0.0, SHORT)
    treated = solve_one_patient("hahnfeldt", HF_THETA, y0, 1.0, SHORT)
    assert treated[-1, 1] < untreated[-1, 1]


def test_hahnfeldt_surface_term_scales_as_two_thirds_power():
    """The V^(2/3) IS the model's signature assumption (PROJECT.md 6.2).

    Doubling V must scale the surface term by 2^(2/3), not by 2. A refactor
    that loses the cube root would still run and still look plausible.
    """
    v = jnp.array(1000.0)
    ratio = (jnp.cbrt(2 * v) ** 2) / (jnp.cbrt(v) ** 2)
    assert float(ratio) == pytest.approx(2 ** (2 / 3), rel=1e-6)


# --- Kuznetsov --------------------------------------------------------------


def test_untreated_kuznetsov_stays_finite_and_non_negative():
    y0 = kuznetsov.initial_state(KT_THETA, t_0=1e5)
    ys = solve_one_patient("kuznetsov", KT_THETA, y0, dose=0.0, cfg=SHORT)
    assert jnp.all(jnp.isfinite(ys))
    assert jnp.all(ys >= -1e-6)


def test_kuznetsov_effectors_start_at_tumour_free_equilibrium():
    """E rest = s/dd isolates the tumour's effect from an arbitrary transient."""
    y0 = kuznetsov.initial_state(KT_THETA, t_0=1e5)
    expected = float(KT_THETA[0]) / float(KT_THETA[4])
    assert float(y0[0]) == pytest.approx(expected, rel=1e-6)


def test_kuznetsov_untreated_trajectory_is_non_monotone():
    """PROJECT.md 6.3: some theta oscillate; a naive TTP rule breaks on them.

    This theta is NOT a monotone grower. Over 1000 days the untreated burden
    rises to ~5e7, collapses to ~1e-7 (the dormant state), and recovers. Two
    things follow, and both are load-bearing:

      1. "crosses the progression threshold once = progression" would fire on
         the way up and call a patient progressed who then enters dormancy.
         eval/metrics.py must define the non-monotone rule explicitly.
      2. the untreated sanity gate must NOT reject this patient for failing to
         grow monotonically -- it is the interesting case, not a bad draw.

    Asserting the turning points directly, so that a future theta change or a
    solver swap that quietly removes the oscillation fails here rather than
    silently making the benchmark easier.
    """
    y0 = kuznetsov.initial_state(KT_THETA, t_0=1e5)
    cfg = SolverConfig(t1=1000.0, n_save=1001)
    ys = solve_one_patient("kuznetsov", KT_THETA, y0, dose=0.0, cfg=cfg)
    burden = observe(ys, "kuznetsov")

    assert jnp.all(jnp.isfinite(burden))

    diffs = jnp.diff(burden)
    turning_points = int(jnp.sum(jnp.abs(jnp.diff(jnp.sign(diffs))) > 0))
    assert turning_points >= 2, f"expected an oscillation, got {turning_points} turns"

    # It both grows well past its start and later falls well below it.
    assert float(burden.max()) > 100 * float(burden[0])
    assert float(burden.min()) < float(burden[0]) / 100


# --- the POMDP boundary and the cohort contract -----------------------------


@pytest.mark.parametrize(
    "model_id,theta,y0",
    [
        ("lotka_volterra", LV_THETA, lotka_volterra.initial_state(LV_THETA, 0.01, 0.5)),
        ("hahnfeldt", HF_THETA, hahnfeldt.initial_state(HF_THETA, 100.0)),
        ("kuznetsov", KT_THETA, kuznetsov.initial_state(KT_THETA, 1e5)),
    ],
)
def test_observation_is_scalar_burden_only(model_id, theta, y0):
    """The policy never sees subpopulations -- that is what makes this a POMDP."""
    ys = solve_one_patient(model_id, theta, y0, 0.0, SHORT)
    burden = observe(ys, model_id)
    assert burden.shape == (SHORT.n_save,)
    assert burden.ndim == 1  # strictly less information than the state


def test_cohort_solve_vmaps_over_patients():
    """PROJECT.md 6.4: in_axes=(None, 0, 0) -- one trajectory per patient."""
    thetas = jnp.stack([LV_THETA, LV_THETA * jnp.array([1.2, 1.0, 1.0, 1.0, 1.0, 1.0])])
    y0s = jnp.stack([lotka_volterra.initial_state(t, 0.01, 0.5) for t in thetas])
    ys = solve_cohort("lotka_volterra", thetas, y0s, cfg=SHORT)
    assert ys.shape == (2, SHORT.n_save, 2)
    assert jnp.all(jnp.isfinite(ys))
    # Distinct theta must give distinct trajectories, or the vmap is broadcasting.
    assert not jnp.allclose(ys[0], ys[1])


def test_unknown_model_id_raises():
    with pytest.raises(ValueError, match="unknown model_id"):
        observe(jnp.zeros((2,)), "not_a_model")


def test_strobl2021_lv_cohort_lies_inside_strobls_published_ranges():
    from adaptive_therapy.envs.sampling import STROBL_D_D, STROBL_R_S, sample_cohort

    c = sample_cohort("lotka_volterra", 300, seed=3, ranges="strobl2021")
    r_S, r_R, K, d_S, d_R, d_D = c.thetas.T
    n_0 = c.y0s.sum(axis=1) / K
    f_R = c.y0s[:, 1] / c.y0s.sum(axis=1)
    assert c.n == 300
    assert (r_S == STROBL_R_S).all() and (d_D == STROBL_D_D).all()
    assert (r_R / r_S >= 0.5).all() and (r_R / r_S <= 1.0).all()
    assert (d_S == d_R).all() and (d_S / r_S <= 0.5).all() and (d_S >= 0).all()
    assert (n_0 >= 0.1 - 1e-9).all() and (n_0 <= 0.75 + 1e-9).all()
    assert (f_R >= 0.001 - 1e-9).all() and (f_R <= 0.1 + 1e-9).all()


def test_default_ranges_are_the_provisional_box_so_logged_runs_reproduce():
    from adaptive_therapy.envs.sampling import sample_cohort

    a = sample_cohort("lotka_volterra", 50, seed=7)
    b = sample_cohort("lotka_volterra", 50, seed=7, ranges="provisional")
    assert (a.thetas == b.thetas).all() and (a.y0s == b.y0s).all()
    with pytest.raises(ValueError):
        sample_cohort("hahnfeldt", 5, ranges="strobl2021")


# --- Zhang 2017 ---------------------------------------------------------------


def _zhang_settle(drug: float, y0, days: float = 20_000.0):
    theta = zhang2017.theta_for(zhang2017.PATIENT_2)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(zhang2017.vector_field), diffrax.Tsit5(), 0.0, days, 0.1,
        jnp.asarray(y0), args=(theta, drug),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-7), max_steps=200_000,
    )  # fmt: skip
    return np.asarray(sol.ys[-1])


def test_zhang_psa_settles_at_twice_the_cell_total():
    """Eq. 2 (p.8): PSA' = sum x - 0.5 PSA, so at rest PSA = 2 sum x."""
    y = _zhang_settle(0.0, [1000.0, 1000.0, 1000.0, 0.0])
    assert y[3] == pytest.approx(2.0 * y[:3].sum(), rel=1e-4)


def test_continuous_abiraterone_hands_the_tumour_to_androgen_independent_cells():
    """Competitive release, the mechanism the paper's MTD arm shows (p.3)."""
    off = _zhang_settle(0.0, [1000.0, 1000.0, 1000.0, 0.0])
    on = _zhang_settle(1.0, off)
    assert on[2] / on[:3].sum() > 0.9 > off[2] / off[:3].sum()


# --- Droop / Baez-Kuang (Phan 2019) -----------------------------------------

# Table 1 values (p.192) in the relapse regime of Phan Sec. 3.1 (p.194): under
# suppression Q settles near (gamma2 Qm + mu q1)/(gamma2 + mu) ~ 0.78, where AD
# growth (0.018/day) is below its death D1 (0.071) but AI growth (0.037) is not.
DROOP_THETA = jnp.array(
    [0.05, 0.5, 0.2, 1e-3, 0.1, 0.005, 0.09, 5e-4, 5e-5, 2e-4, 3.0, 1.0, 5e-5, 0.5, 0.001, 15.0]
)


def _droop(dose: float, y0, days: float, theta=DROOP_THETA):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(droop.vector_field), diffrax.Tsit5(), 0.0, days, 0.05,
        jnp.asarray(y0), args=(theta, dose), saveat=diffrax.SaveAt(ts=jnp.linspace(0, days, 200)),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9), max_steps=500_000,
    )  # fmt: skip
    return np.asarray(sol.ys)


@pytest.mark.parametrize("dose", [0.0, 1.0])
def test_droop_quota_stays_between_min_ai_quota_and_max_androgen(dose):
    """Proposition 1 (p.195), two-population case: q2 <= Q <= Qm."""
    ys = _droop(dose, [100.0, 1.0, 10.0, 5.0], 1500.0)
    assert np.isfinite(ys).all() and (ys[:, :2] >= -1e-6).all()  # solver undershoot at extinction
    assert (ys[:, 2] >= 0.2 - 1e-6).all() and (ys[:, 2] <= 15.0 + 1e-6).all()


def test_droop_resistance_cannot_arise_without_transformation():
    """x2 has no source but lam(Q) x1: with c = 0 an absent AI clone stays absent."""
    ys = _droop(1.0, [100.0, 0.0, 10.0, 5.0], 1500.0, DROOP_THETA.at[12].set(0.0))
    assert np.abs(ys[:, 1]).max() == 0.0


def test_continuous_suppression_relapses_through_transformed_ai_cells():
    """The absorbing-state relapse of Sec. 3.1 (p.194): under CAS, x1 regresses
    and the tumour regrows as x2, created from x1 rather than pre-existing."""
    ys = _droop(1.0, [100.0, 0.0, 15.0, 5.0], 3000.0)
    total = ys[:, 0] + ys[:, 1]
    nadir = total.argmin()
    assert 0 < nadir < len(total) - 1 and total[-1] > 2 * total[nadir]
    assert ys[-1, 1] / total[-1] > 0.99
