# Paper-to-implementation audit

Target: the 45-page *Tracking Drift* paper, arXiv:2601.19624, especially
Sections 4.1–4.2 and Appendix C. This revision does not assert that its new
runs reproduce the published table values.

| Paper requirement | Implementation and status |
|---|---|
| Eq. (274), joint absolute critic residual quantile | SAC/flow scheduler input; SQL uses its soft-Q residual |
| Eq. (275), PPO value TD deltas | Uses one-step rollout bootstrap targets minus the current minibatch value; formerly used GAE-return regression errors |
| Eqs. (276)–(278), EMA and prefix-mean square root | Scheduler checked against hand calculations; rejects invalid inputs; portable/Isaac schedulers serialize state |
| SAC automatic-temperature baseline | Portable SAC already autotunes; Isaac entry point now enables autotuning for baseline only |
| Same alpha in actor and soft backup | SAC recomputes the target algebraically after scheduling; flow synchronizes online and target alpha |
| SQL carrier | New continuous-action SQL with bounded-action soft integral and amortized SVGD; not the missing original implementation |
| 5 patterns × 5 seeds × 12 tasks × 4 carriers × 2 variants | Complete 2,400-cell manifest; Toy/MuJoCo executable, Isaac cells explicitly blocked |
| Progress-aligned changes | Environment-interaction index is passed explicitly; float boundary rounding fixed; no extra abrupt event at T |
| Toy abrupt displacement 0.5 | Alternates between offsets 0 and 0.5, fixing the previous later jumps of 1.0 |
| Isaac gravity/damping perturbations | No silent stationary fallback; unsupported simulator hooks raise before training |
| 10 deterministic evaluation episodes | Portable carriers use deterministic actions at 0, exact evaluation intervals and T; SAC no longer samples evaluation actions |
| Eq. (32), AUC integral | Actual trapezoidal integral; previous script returned a time average |
| Eq. (33), nAUC | Family Steady SAC denominator, equal-weight task means |
| Eq. (34), drop ratio | Same task/carrier/variant/seed Steady reference |
| Eq. (35), recovery | First observed return at/above the last pre-change observation; sum of delays / T |
| Reconstructable results | Raw evaluation JSONL, arguments, versions, weights, scheduler state and CSV summaries |

## Choices the paper does not uniquely determine

- Original per-task horizons, all carrier hyperparameters, drift RNG seed,
  Mixed formula, interpolation endpoint convention, and exact SQL settings are
  not fully recoverable. The manifest exposes the horizon; existing deterministic
  drift formulas are retained except the demonstrated boundary and displacement
  bugs. Linear drift still resets within each segment, as the released code did.
- PPO: Appendix C.4 calls the proxy a rollout value residual; C.4/C.5 also specify
  the current update batch and gradient-update clock. We cache rollout TD(0)
  bootstrap targets and subtract current minibatch values, updating AES once per
  optimization minibatch. GAE remains the policy-advantage estimator only.
- Recovery is censored at the next change; the paper does not say how to report
  non-recovery. We retain null and a censoring count, rather than call it zero or
  silently exclude it from a family mean. Resolution is limited by evaluation
  spacing; there is no invented sub-grid recovery time.
- Family normalization is performed by the mean Steady SAC AUC across tasks and
  seeds, followed by equal-weight task aggregation. Different undocumented
  normalization/seed-pairing choices can change table values.
- SQL uses a single soft-Q critic, 64 uniform value particles, 16 SVGD particles,
  and temperature 1.0 for its fixed baseline. These are stated reconstruction
  choices. Its deterministic action is a mean over a fixed antithetic particle
  bank, an approximation for an implicit policy. Do not label it the original
  SQL run. Algorithm reference: [Haarnoja et al., 2017](https://proceedings.mlr.press/v70/haarnoja17a.html).
- The original Toy environment returns mostly negative rewards. With negative
  reference AUC, a larger paper ratio need not mean better return. The reporter
  emits the literal formula and its validity flag. An undocumented reward offset
  must not be introduced to manufacture the paper's ordering.

## Remaining blockers to full numerical reproduction

1. Original per-task configurations and raw per-seed learning/evaluation curves,
   including the exact normalization reference and checkpoints.
2. Original SQL implementation/configuration to replace or validate the new
   reconstruction; confirmation that the released flow carrier is the exact
   MEow variant used for all reported tasks.
3. An installed compatible Isaac simulator, task-specific simulator-backed
   gravity/damping adapters, and the original interaction-count convention.
   Merely changing a Python attribute is not proof of a physics change.
4. Isaac PPO/SQL carriers and a common 10-episode deterministic evaluation
   pipeline. Legacy Isaac training reward logs cannot be substituted for these.
5. Full five-seed training and result comparisons. CPU smoke runs do not establish
   the paper's performance gains or A100 timings.

`final.pt` is a portable model/scheduler snapshot, not an exact simulator/replay
restart. It deliberately makes no resume-equivalence claim. Checkpoints do not
contain serialized live model objects.
