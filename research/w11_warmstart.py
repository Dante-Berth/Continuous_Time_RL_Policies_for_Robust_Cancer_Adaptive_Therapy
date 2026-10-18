"""W11: is the recurrent deficit a representation limit or a PQN learning cost?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-18] W11, before any cell
ran. Every arm is first behaviour-cloned from a threshold teacher (Stage A),
then PQN fine-tuned from that policy (Stage B). S1 Part A's cohort, seeds
31-40, Strobl ranges, w in {0.05, 0.1, 0.2, 0.4}, S1's arm sizes.

  T_fix  one threshold per (w, seed): O1b's leave-one-seed-out level
  T_pat  one threshold per training patient: the level (O1's grid) with the
         best U on that patient's own noisy training rollout

The teacher labels the student's OWN observation (the noisy log-burden both
see), so T_fix is a function of the current observation and T_pat needs the
patient's level, which a burden-only student can only infer from history.

Stage A: teacher rollouts, then 3 DAgger rounds (student greedy rollouts
relabelled by the teacher); 2000 Adam(3e-4) steps in all, 500 per stage, on
minibatches of 32 whole episodes drawn from the aggregate. Stage B: PQN as
S1, 300 updates, epsilon fixed at 0.005.

Run: JAX_PLATFORMS=cpu uv run python research/w11_warmstart.py
     -> research/outputs/w11_warmstart.json, w11_shards/, w11_policies/
Smoke: W11_SMOKE=1 (tiny budgets, one seed, one w; writes to w11_smoke_*).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o1_oracle as o1  # noqa: E402  (imports s1_strobl: backend env first)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.policies.base import N_ACTIONS  # noqa: E402
from adaptive_therapy.policies.rules import ThresholdPolicy  # noqa: E402
from adaptive_therapy.rl import pqn  # noqa: E402
from adaptive_therapy.rl.rollout import batch_rollout, replay, rollout  # noqa: E402

s1, w4, g, wf = o1.s1, o1.w4, o1.g, o1.s1.wf
SMOKE = bool(os.environ.get("W11_SMOKE"))
TAG = "w11_smoke" if SMOKE else "w11"
SEEDS = (31,) if SMOKE else s1.SEEDS_A
WEIGHTS = (0.1,) if SMOKE else w4.WEIGHTS
ARCHS = s1.ARCHS_A
TEACHERS = ("fix", "pat")
TIERS = s1.TIERS
STAGES, STEPS_PER_STAGE = 4, (10 if SMOKE else 500)  # teacher rollouts + 3 DAgger rounds
BC_LR, BC_BATCH = 3e-4, 32
TAU = 0.1  # logit temperature of the cloning loss; keeps outputs on a Q-like scale
N_UPDATES = 3 if SMOKE else 300
GATE_D = 10.0  # days, as registered
MAX_NONFINITE = 0.05
SHARDS = g.OUT / f"{TAG}_shards"
POLICIES = g.OUT / f"{TAG}_policies"
RESULT = g.OUT / f"{TAG}_warmstart.json"
O1B = json.loads((g.OUT / "o1b_split.json").read_text())["chosen_levels"]


def teacher_action(level, obs):
    """The threshold rule on the student's own observation (policies/rules.py)."""
    return jnp.where(obs > jnp.log(level), N_ACTIONS - 1, 0)


def train_cohort(seed: int):
    c = sample_cohort(g.FAM, g.N_TRAIN, seed=seed, ranges=g.RANGES)
    return jnp.asarray(c.thetas), jnp.asarray(c.y0s)


def patient_levels(th, y0, w: float) -> np.ndarray:
    """T_pat: per patient, the grid level with the best U on its own noisy rollout."""
    rw = RewardConfig()
    us = []
    for x in o1.LEVELS:
        tr = batch_rollout(ThresholdPolicy(float(x)), th, y0, o1.KEY, g.EP, rw, greedy=True)
        us.append(o1.u_of(o1.endpoints(tr["burden"], tr["dose"]), w))
    return o1.LEVELS[np.argmax(np.stack(us), axis=0)].astype(np.float32)


def teacher_levels(teacher: str, th, y0, w: float, seed: int) -> jnp.ndarray:
    if teacher == "fix":
        return jnp.full(th.shape[0], O1B[f"{w}|{seed}"], dtype=jnp.float32)
    return jnp.asarray(patient_levels(th, y0, w))


def teacher_rollout(levels, th, y0, key, rw):
    keys = jax.random.split(key, th.shape[0])
    return jax.vmap(
        lambda t, y, k, lv: rollout(ThresholdPolicy(lv), t, y, k, g.EP, rw, greedy=True)
    )(th, y0, keys, levels)


def behaviour_clone(policy, th, y0, levels, key, w: float):
    """Stage A. Returns (policy, final loss, final teacher-agreement on the aggregate)."""
    rw = RewardConfig(exposure_penalty=w)
    opt = optax.adam(BC_LR)
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))
    ep = replace(g.EP, drop_rate=0.0)

    def loss_fn(pol, obs, lv):
        logits = jax.vmap(lambda o: replay(pol, o, ep)[0])(obs)
        labels = teacher_action(lv[:, None], obs)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits / TAU, labels)
        return ce.mean(), (jnp.argmax(logits, -1) == labels).mean()

    @eqx.filter_jit
    def step(pol, opt_state, obs, lv):
        (loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(pol, obs, lv)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(pol, eqx.is_array))
        return eqx.apply_updates(pol, updates), opt_state, loss, acc

    obs_bank, lv_bank = [], []
    loss = acc = jnp.nan
    for stage in range(STAGES):
        key, k_roll, k_sgd = jax.random.split(key, 3)
        if stage == 0:
            traj = teacher_rollout(levels, th, y0, k_roll, rw)
        else:
            traj = batch_rollout(policy, th, y0, k_roll, g.EP, rw, greedy=True)
        obs_bank.append(traj["obs"])
        lv_bank.append(levels)
        obs_all, lv_all = jnp.concatenate(obs_bank), jnp.concatenate(lv_bank)
        for k in jax.random.split(k_sgd, STEPS_PER_STAGE):
            idx = jax.random.choice(k, obs_all.shape[0], (BC_BATCH,), replace=False)
            policy, opt_state, loss, acc = step(policy, opt_state, obs_all[idx], lv_all[idx])
    return policy, float(loss), float(acc)


def cell(arch: str, teacher: str, w: float, seed: int) -> dict:
    k_init, k_train, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
    th, y0 = train_cohort(seed)
    levels = teacher_levels(teacher, th, y0, w, seed)
    t0 = time.perf_counter()
    policy, bc_loss, bc_acc = behaviour_clone(
        w4.ARCHS[arch](k_init), th, y0, levels, jax.random.fold_in(k_train, 1), w
    )
    t_bc = time.perf_counter() - t0
    w4.save_policy(policy, POLICIES / "bc", f"{arch}_{teacher}", w, seed)
    out = {
        "arch": arch,
        "teacher": teacher,
        "w": w,
        "seed": seed,
        "bc_loss": bc_loss,
        "bc_agreement": bc_acc,
        "bc_seconds": t_bc,
        "bc": s1.tiers_eval(policy, seed, w),
    }
    cfg = pqn.PQNConfig(n_updates=N_UPDATES, eps_start=0.005, eps_finish=0.005)
    t0 = time.perf_counter()
    policy, hist = pqn.train(
        policy, th, y0, k_train, cfg, replace(g.EP, drop_rate=0.0), RewardConfig(exposure_penalty=w)
    )
    out["ft_seconds"] = time.perf_counter() - t0
    out["ft_history"] = hist
    w4.save_policy(policy, POLICIES / "ft", f"{arch}_{teacher}", w, seed)
    out["ft"] = s1.tiers_eval(policy, seed, w)
    return out


def teacher_ref(teacher: str, w: float, seed: int) -> dict:
    """The teacher scored on the test cohorts. T_pat picks, per TEST patient, the best
    grid level on the scored rollout itself (O1's 'tuned' convention): a ceiling, not
    a deployable policy, and optimistic by selection on the evaluation noise."""
    out = {}
    for tier, cohort in (("tier0", w4.tier0_cohort), ("tier1", g.tier1_cohort)):
        evs = [
            w4.evaluate(ThresholdPolicy(float(x)), seed, w, cohort=cohort, drops=(0.0,))["0.0"]
            for x in (o1.LEVELS if teacher == "pat" else [O1B[f"{w}|{seed}"]])
        ]
        if teacher == "fix":
            out[tier] = evs[0]
            continue
        u = np.stack([w4.utility({"0.0": e}, 0.0, w) for e in evs])  # (levels, patients)
        j = np.argmax(u, axis=0)
        out[tier] = {
            k: np.stack([np.asarray(e[k]) for e in evs])[j, np.arange(len(j))].tolist()
            for k in ("ttp_capped", "censored", "exposure", "burden_integral")
        }
    return out


def spawn(arch: str, teacher: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{arch}_{teacher}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", arch, teacher, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
            env={**os.environ},
        )
    print(f"  done {arch:8s} {teacher} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


# --- analysis ---------------------------------------------------------------------------


def arm_cells(shards: list[dict]) -> list[dict]:
    """Shards -> w4-style cells named bc_<arch>_<teacher> / ft_<arch>_<teacher>."""
    out = []
    for s in shards:
        for stage in ("bc", "ft"):
            out.append(
                {
                    "arch": f"{stage}_{s['arch']}_{s['teacher']}",
                    "w": s["w"],
                    "seed": s["seed"],
                    "closed_loop": s[stage]["closed_loop"],
                    **{t: s[stage][t] for t in TIERS},
                }
            )
    return out


def cmp(cells, valid, a, b, tier) -> dict:
    return w4.comparison(cells, valid, a, b, "level", SEEDS, (tier,))


def did(cells, valid, arch, teacher, tier) -> dict:
    """(ft - bc)[arch] - (ft - bc)[mlp], paired on patient, seed+patient bootstrap."""
    d = w4.paired(cells, valid, f"ft_{arch}_{teacher}", f"bc_{arch}_{teacher}", "level", SEEDS, (tier,))
    m = w4.paired(cells, valid, f"ft_mlp_{teacher}", f"bc_mlp_{teacher}", "level", SEEDS, (tier,))
    return w4.delta_ci(d - m, (tier,))


def reading_p1(r: dict) -> str:
    lo, hi = r["ci95"]
    return "memory helps" if lo > GATE_D else "ties" if hi < GATE_D else "inconclusive"


def reading_p2(r: dict) -> str:
    lo, hi = r["ci95"]
    return "learning cost" if (r["delta"] <= -GATE_D and hi < 0) else "no differential cost"


def analyse(shards: list[dict]) -> dict:
    cells = arm_cells(shards)
    # W4's helpers read global WEIGHTS; the smoke run uses a subset
    w4.WEIGHTS = WEIGHTS
    refs = {
        (t, w, s): {"arch": f"T_{t}", "w": w, "seed": s, **teacher_ref(t, w, s)}
        for t in TEACHERS
        for w in WEIGHTS
        for s in SEEDS
    }
    cells += list(refs.values())
    valid = {(c["arch"], c["w"], c["seed"]): True for c in cells}
    for c in cells:
        ok = all(np.isfinite(c[t][k]).all() for t in TIERS for k in ("ttp_capped", "exposure"))
        valid[(c["arch"], c["w"], c["seed"])] = bool(ok)
    names = sorted({c["arch"] for c in cells})
    res = {"prereg": "adaptive_therapy/PREREG.md W11 [2026-10-18]", "smoke": SMOKE}
    res["nonfinite"] = [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok]
    res["arm_valid"] = {
        a: sum(not valid[(a, w, s)] for w in WEIGHTS for s in SEEDS) / (len(WEIGHTS) * len(SEEDS))
        <= MAX_NONFINITE
        for a in names
    }
    res["closed_loop"] = {
        a: [c["closed_loop"] for c in cells if c["arch"] == a and "closed_loop" in c]
        for a in names
        if a.startswith(("bc_", "ft_"))
    }
    res["bc_agreement"] = {
        f"{a}_{t}": [s["bc_agreement"] for s in shards if s["arch"] == a and s["teacher"] == t]
        for a in ARCHS
        for t in TEACHERS
    }
    # gate: the cloned MLP reaches T_fix within 10 d
    gate = cmp(cells, valid, "bc_mlp_fix", "T_fix", "tier0")
    res["gate"] = {**gate, "passes": bool(gate["ci95"][0] > -GATE_D)}
    res["P1"], res["P2"], res["vs_teacher"] = {}, {}, {}
    for tier in TIERS:
        for t in TEACHERS:
            for a in ("lstm", "ode_rnn"):
                r = cmp(cells, valid, f"bc_{a}_{t}", f"bc_mlp_{t}", tier)
                res["P1"][f"{a}_{t}_{tier}"] = {**r, "reading": reading_p1(r)}
                r = did(cells, valid, a, t, tier)
                res["P2"][f"{a}_{t}_{tier}"] = {**r, "reading": reading_p2(r)}
            for a in ARCHS:
                res["vs_teacher"][f"bc_{a}_{t}_{tier}"] = cmp(cells, valid, f"bc_{a}_{t}", f"T_{t}", tier)
                res["vs_teacher"][f"ft_{a}_{t}_{tier}"] = cmp(cells, valid, f"ft_{a}_{t}", f"T_{t}", tier)
                res["vs_teacher"][f"change_{a}_{t}_{tier}"] = cmp(
                    cells, valid, f"ft_{a}_{t}", f"bc_{a}_{t}", tier
                )
    res["wilcoxon"] = {
        f"{a}_{t}": w4.wilcoxon_per_w_p(cells, valid, f"bc_{a}_{t}", f"bc_mlp_{t}", SEEDS, ("tier0",))
        for a in ("lstm", "ode_rnn")
        for t in TEACHERS
    }
    res["table"] = w4.table(cells, valid, [], names, SEEDS, TIERS)
    return res


def ci(v: dict) -> str:
    return f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"


def report(res: dict) -> None:
    print(f"nonfinite {res['nonfinite']}  arm_valid {res['arm_valid']}")
    print(f"GATE (bc_mlp_fix - T_fix, Tier 0, lower CI > -{GATE_D:.0f}): {ci(res['gate'])}  PASS={res['gate']['passes']}")
    for k, v in res["P1"].items():
        print(f"  P1 {k:22s} {ci(v)}  {v['reading']}")
    for k, v in res["P2"].items():
        print(f"  P2 {k:22s} {ci(v)}  {v['reading']}")
    for k, v in res["vs_teacher"].items():
        if k.endswith("tier0"):
            print(f"  {k:28s} {ci(v)}")
    for a, cl in res["closed_loop"].items():
        print(f"  closed-loop {a:18s} median {np.median(cl):.3f}  <0.10: {np.mean(np.array(cl) < 0.10):.0%}")


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, t, w, s) for a in ARCHS for t in TEACHERS for w in WEIGHTS for s in SEEDS]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        shards = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(shards)
    RESULT.write_text(json.dumps(res, indent=2))
    report(res)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, teacher, w, seed = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        SHARDS.mkdir(parents=True, exist_ok=True)
        (SHARDS / f"{arch}_{teacher}_w{w}_s{seed}.json").write_text(
            json.dumps(cell(arch, teacher, w, seed))
        )
        raise SystemExit(0)
    raise SystemExit(main())
