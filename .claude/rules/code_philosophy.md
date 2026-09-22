# Code philosophy — always applies

## 1. Stack & tooling

- **Python, managed with `uv`.** Every script runs via `uv run`; `make setup`
  is the only bootstrap. Dependencies are declared in `[project.dependencies]`
  and locked in a committed `uv.lock` — not in per-module requirements files.
- **Math: JAX + equinox + diffrax + optax**, exclusively. No pure-Python loops
  for anything vectorisable. A network is a pytree and drops straight into a
  `diffrax` vector field — that is why this stack was chosen over Flax-linen.
- **Automation: the Makefile.** `make all` = lint + test + audit + lit-check.
  If a command is worth running twice, it is worth a target.

## 2. Engineering philosophy

- **Minimalism.** The simplest thing that answers the question. A workshop paper
  is 4–9 pages; architectural pile-up (MoE + liquid + meta + PINN at once) reads
  as insecurity to a reviewer.
- **Strict architectural adherence.** Respect the existing tree and naming. New
  code blends in; do not "innovate" structurally.
- **One home per piece of logic.** A second copy of a TTP calculation or a
  solver is a bug, not a convenience. Import it.
- **`adaptive_therapy/` is imported and tested; `research/` is one-shot
  scripts.** All I/O lives in `research/`, never in the package. Artifacts go to
  gitignored `research/outputs/`.
- **Three similar patterns → consider extracting. Two → leave them**; the
  abstraction cost is not paid yet.

## 3. Testing

- Every new behaviour ships with a test. **Test names state the invariant, not
  the function** — `test_untreated_lv_is_bounded_by_carrying_capacity`, not
  `test_lotka_volterra`.
- 100% of tests pass, new and pre-existing, before any commit.
- Prefer property/recovery tests: parameter recovery against a simulated
  trajectory; a known analytic solution; conservation under a null treatment.

## 4. Research-code specifics

- Guard divisions with `jnp.maximum(x, 1e-10)`, never `x + 1e-10`.
- `jnp.isfinite(...).all()` before trusting a rollout or a gradient.
- **`vmap` over an adaptive-step solver costs the stiffest patient's step count
  for every patient.** Fixed-step for bulk rollouts; bucket by stiffness;
  `Kvaerno5` only for the stiff bucket. Measure before believing a timing claim.
- Clamp θ to plausible ranges under large shifts; lower the learning rate before
  blaming the architecture for NaNs.
- Seeds threaded explicitly (`seed: int = SEED`); never global seeding.
- **Dated, evidenced comments on every config exception** — *why*, with the
  incident that produced it. This is the most transferable habit in the sibling
  repos and the cheapest thing to lose.
