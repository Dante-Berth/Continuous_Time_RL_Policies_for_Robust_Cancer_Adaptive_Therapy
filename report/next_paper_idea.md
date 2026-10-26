# Idea for a second paper: does the objective, not the network, decide how much drug a controller uses?

Status: an idea, not a result. Nothing here has been run. It is not part of the
research record; if we pursue it, it must first be registered in PREREG.md.

Target: arXiv preprint (no page limit and no reviewers, so the registration and
the controls are the only quality check; keep them strict).

## 1. The main idea in three sentences

In our first paper we asked if a special network (an ODE-RNN) can plan cancer
treatment better than a simple network. The answer was no, and we also found
that a controller that can see the hidden state of the tumour did not beat a
simple controller that sees only the tumour size, under four training methods.
In the data we noticed one strange detail: the controller with full information
reaches a smaller tumour burden, but only because it uses more drug for some
patients. The new paper asks: **is the problem the training objective, and can
a better objective make a controller that uses less drug for the same control?**

## 2. Background (what the reader must know)

- **Adaptive therapy.** Doctors do not try to remove every cancer cell. They
  keep some drug-sensitive cells alive, because these cells compete with the
  drug-resistant cells and slow them down. The drug is given, paused and given
  again, depending on the size of the tumour.
- **Virtual patients.** We cannot test many treatment rules on real people, so
  we use computer models of tumours (three families of equations) with many
  different parameter values. Each parameter set is one virtual patient.
  We make no claim about real patients.
- **Partial observation.** The controller sees only the total tumour burden.
  It does not see how many cells are sensitive and how many are resistant.
  This is the same situation as in a real clinic.
- **Score.** We measure time to progression (TTP, capped at 357 days) and the
  total drug used (exposure). A long TTP is not good if it needs constant drug,
  so we combine them: score = min(TTP, 357) - w x exposure, with a weight w.
- **Reference controller.** To know if a task is learnable, we train one
  controller that can see the hidden state. If it cannot beat the simple
  controller, there is no room for memory to help, and a comparison of
  architectures says nothing. This is our admission rule.

## 3. What we found in paper 1 that leads to this idea

1. Four training methods (PPO, PQN, recurrent replay Q-learning, evolution
   strategies) all failed the admission rule. The reference did not beat the
   simple controller on the capped score.
2. We then asked if the score itself was unfair to the reference. We measured
   the area under the burden curve (the burden integral) from the saved runs.
3. The reference did have a smaller burden integral: about 17 to 20 percent at
   Tier 0. But it used more drug on average: +50 days of cumulative exposure
   under one run and +22 days under another. The median dose was the same, so
   the extra drug went to a minority of patients (a tail).
4. With PPO the result was the opposite: the reference had a larger burden and
   used less drug.

So the reference is not simply "better" or "worse". It makes a different
trade between burden and drug, and the trade changes with the training method.

## 4. The new question and why it matters

**Question.** When a controller sees more information, does it use that
information to reduce the drug, or to reduce the burden at the cost of more
drug? And which part of this choice comes from the objective we give it?

Why it matters:

- If the objective decides the trade, then comparing controllers with one fixed
  objective can hide real differences. Two controllers may be equally good but
  placed at different points of the same burden-drug curve.
- A clinic wants a rule that controls the tumour with little drug. A controller
  that gains control by heavy dosing in a few patients is not acceptable, even
  if its average looks good.
- It would give a better way to judge controllers than a single score.

## 5. Proposed approach

The comparison is a **trade-off curve**, not a single number.

1. **Train a family of controllers.** For each of several values of w (the drug
   weight), train the reference and the simple controller. Use one training
   method only, chosen before the runs (the one with the most stable gate
   result in paper 1; the choice must be written in the registration).
2. **Draw the frontier.** For each controller, plot the median burden integral
   against the median exposure over the same virtual patients. Show the
   interquartile range and use several seeds.
3. **Compare the frontiers.** Ask whether the reference frontier lies below the
   simple-controller frontier (less burden at equal drug). This is a fairer
   test than one score with one w.
4. **Look at the tail.** Report how much drug the top 10 percent of patients
   receive. Test one change: a penalty on the worst-case dose (for example, a
   limit on the drug in any 28-day window) and see if the reference still
   gains.
5. **Keep the controls from paper 1:** burden-only observation for the simple
   controller, paired tests across matched patients, and exposure always
   reported with TTP.

## 6. Hypotheses (to be registered before any run)

- H1. At equal exposure, the reference frontier has a smaller burden integral
  than the simple-controller frontier, in both Tier 0 and Tier 1.
- H2. The extra exposure seen in paper 1 is mostly a tail effect: removing the
  top 10 percent of patients by dose removes most of the difference.
- H3. A dose-window limit reduces the tail without removing the gain of the
  reference.

We also write what would count as failure: if the frontiers overlap within the
seed variation, the answer is "the objective does not hide an advantage", and
that is a valid negative result. Success and failure rules must be written with
numbers before the run.

## 7. Possible outcomes and what each would mean

| Outcome | Meaning |
|---|---|
| Reference frontier is lower | Information helps; paper 1's gate used a score that hid it. Architectures can then be compared on the frontier. |
| Frontiers overlap | The simple controller is already near the best possible. The memory question is closed for this simulator. |
| Gain only in the tail | The advantage is a risk trade, not a real improvement; report it as a warning for other studies. |

## 8. Risks and limits

- It uses the same simulators as paper 1. A finding may be only about these
  virtual patients. We say this clearly and make no clinical claim.
- Training for many values of w costs much more computing time. Plan the
  budget before registering.
- It could look like "one more attempt after four failures". The registration
  must say that this is a new question about the evaluation, and it must not
  reopen the closed trainer question (CLOSED #1).
- The strongest form of the idea (a new loss for the dose tail) is a larger
  project. Start with the evaluation-only version (steps 1-3), because it needs
  no new training method.

## 9. Possible structure of the paper

1. Introduction: why one score can hide a trade-off.
2. Background: adaptive therapy, partial observation, the admission rule.
3. Method: the frontier, the tail measures, the dose-window limit.
4. Experiments: Tier 0, Tier 1, several seeds, paired tests.
5. Results: frontiers, tail analysis, effect of the limit.
6. Discussion: what a clinic or a later study should report.
7. Limits: virtual patients only.

## 10. First step if you agree

Write the registration (PREREG.md) with the exact training method, the list of
w values, the number of seeds, the pass and failure numbers, and the compute
budget. Run nothing before it is committed.
