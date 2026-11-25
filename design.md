# Design: what the project shows and the checks

## Question

A recommender is retrained every day on a window of its own logs. A change that alters the data the product
generates, here more exploration, acts twice: directly on what users see now (exploration costs clicks today) and
through the data the next models train on (new good items are found sooner). Why do standard A/B tests reject such
a change, and does a natural fix recover its long-run value?

This is a known problem (see the related work in `docs/readout.md`). The project claims no new method. It shows the
problem in a small simulation where the true launch effect is known, and it reports an honest negative result for one
fix.

## Where the results come from

The results were first obtained in a pre-registered research check: three kill tests (KT1 for the four
designs below, KT2 and KT3 for variants of the fix), each with a positive control run first. The verdicts were KT1
MARGINAL, KT2 MARGINAL and KT3 KILL, and the research line was stopped. This project repackages that work. It
reruns the same simulation code with the same seeds, checks that every number reproduces (P1, P2), and presents the
results in two figures and a short readout. Nothing in the world, the designs, the seeds or the scoring rules is
changed.

## The toy world (parameters as in the source)

- K = 200 items. Each day a share rho = 0.05 of the items is replaced by new items (each item independently with
  probability 0.05). Item click rates are drawn from Beta(1, 19), mean 0.05.
- Each day 2V impressions, V = 5,000 per half of the traffic. Each impression shows one item and is a click with the
  item's rate.
- A model is a table of per-item clicks and impressions over the last W days of its training log. Its score for an
  item is the posterior mean under a Beta(1, 19) prior, so an unseen item scores 0.05. Every model is retrained once
  a day and trains on V impressions per day of log (a subsample when more are available), so the amount of training
  data is the same in every design and at launch.
- Serving policy of arm a: with probability eps_a show a uniformly random item, otherwise the item with the highest
  score. Control eps_0 = 0.01; treatment eps_1 in {0.05, 0.15}.
- Training window W in {28, 56} days. With the two treatment levels this gives 4 cells.
- Outcome: click. Launch effect Delta = the steady-state click rate when every impression is served by the treatment
  policy and every model trains on treatment logs, minus the same for control. Truth from 200 independent runs per
  arm after a burn-in of 3W days, averaged over the next 60 days, with its Monte Carlo SE.

## The four designs

Every test starts from a snapshot of a control-only world after a burn-in of 3W days (so the window holds W days of
control logs) and lasts T = 14 days, with traffic split 50/50 between the serving arms.

1. Pooled A/B: one model trained on the pooled log of both arms (V per day subsampled) serves both arms. Estimate:
   difference in mean daily click rate between arms over the test.
2. Data-diverted A/B: each arm's model trains on its own arm's test-period logs plus the shared pre-test logs. Same
   estimate.
3. Switchback: all traffic follows one arm per day, 7 days per arm in a random order, one model on the pooled log.
   Estimate: mean click rate of treated days minus control days.
4. Share-randomized replicas (the fix): three model replicas whose test-period training days hold a share
   f in {0, 1/2, 1} of treated logs (f V impressions from arm 1, (1 - f) V from arm 0); users are randomized to a
   serving arm (1/2 each) and a replica (1/3 each). On test day d the replica's treated share of its window is
   s = f (d - 1) / W, at most 13/W (0.46 at W = 28, 0.23 at W = 56). Per arm, least squares of daily click rate on s;
   launch prediction alpha_1 + beta_1 - alpha_0, a linear extrapolation to s = 1.

Variants of the fix (KT2, KT3 in the source), same world and cells:
- S2: two replicas, f in {0, 1}, T = 14. S2L: the same with T = 28 days. Linear extrapolation as above.
- Learning curve: S2 with arm 1 fitted as y = alpha_1 + b (1 - exp(-n / lambda)), n = s W treated days in the window,
  lambda on a grid of 60 log-spaced values from 0.5 to 500 days; launch = alpha_1 + b (1 - exp(-W / lambda)) - alpha_0.
- Short window (SW): two replicas trained only on test-period logs (f = 0 or 1), T = 14; estimate = click rate of
  (arm 1, f = 1) minus (arm 0, f = 0) over the last 7 test days.
- Reported: S2 with 10 times the traffic (V = 50,000) in cell (W = 28, eps_1 = 0.05). The replicas still train on V =
  5,000 impressions a day, so only the user-sampling noise of the evaluation shrinks.

Score per design and cell: the median over 100 tests of the relative error (estimate - Delta) / |Delta|, with a
bootstrap 95% interval (2,000 resamples), and the noise ratio SD(estimate) / |Delta|. A relative error of -1 means the
test sees no effect at all; below -1 it has the wrong sign.

## Positive controls

The same code with the click model replaced by a linear state model: an impression in arm a served by a model whose
window has treated share s is a click with probability 0.05 + tau a + gamma s (tau = -0.004, gamma = 0.008, so
Delta = 0.004). The recommender only produces the logs. Criteria, as registered in the source:
- four designs (cell W = 28, eps_1 = 0.05): the fix's mean error within 3 Monte Carlo SEs of 0 and below 5% of Delta;
  the data-diverted design recovers tau + gamma kappa with kappa within 0.05 of the computed mean treated share of the
  treated arm's window (6.5 / 28 = 0.232); pooled and switchback within 0.1 |gamma| of tau;
- variants: S2 and S2L linear mean errors within 3 Monte Carlo SEs of 0 over 2,000 tests.

## Checks

- **P1, positive control.** (a) The four-design control is run with 2,000 tests, not the source's 10,000 (an allowed
  difference, to keep the whole run under 10 minutes). Tests are seeded by their index, so these are the source's
  first 2,000 tests: every stored number must be identical to the source file to 6 decimals (absolute difference below
  5e-7), and the source's criteria above must hold on them. With 2,000 tests the 5% bound is a weak check (Monte Carlo
  SE of the mean about 0.00013 against a bound of 0.0002); the identity with the source run, whose 10,000 tests met the
  bound, carries the weight. When this plan was written, the source file's first 2,000 tests gave a fix error of
  +0.000022 (0.18 SE), kappa 0.234, pooled -0.00401 and switchback -0.00413, so the criteria are known to be
  reachable. (b) The variants' control with the source's 2,000 tests: identical numbers to 6 decimals, the criterion
  above, and a score file identical to the source's.
- **P2, main results.** A rerun with the source's seeds (100 tests per design and cell) gives every stored number of
  KT1 and KT2/KT3 (Delta, its SE, every test's estimate) identical to the source files to 6 decimals, and score files
  (medians, bootstrap intervals, SDs, noise ratios, verdicts, with the source's scoring rules and bootstrap seeds)
  identical in text to the source's. The verdicts must read KT1 MARGINAL, KT2 MARGINAL, KT3 primary KILL, KT3
  secondary KILL.
- **P3, figures from files.** `scripts/make_figures.py` reads only `results/*.json` and imports no simulation code; a
  second run gives byte-identical PNG files.

`scripts/check_reproduction.py` runs P1 and P2 against the source result files and prints PASS or FAIL with the
numbers. The run sets OPENBLAS_NUM_THREADS=1 and should take under 10 minutes on a laptop.

## Figures and readout

- (a) Median relative error by design and cell, with the middle half of the 100 tests: the wrong sign of pooled tests
  and switchbacks, the partial view of data-diverted tests and the spread of the fix.
- (b) The fix's estimates at 1x and 10x traffic in one cell: trajectory noise that more users do not reduce.
- `docs/readout.md`: the decision problem, what each design estimates, the results, why the fix fails, what a team
  can do instead, related work.
