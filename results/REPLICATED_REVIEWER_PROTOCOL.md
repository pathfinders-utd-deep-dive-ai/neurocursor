# Strengthened reviewer controls

These are retrospective development experiments on the existing dataset, not
external confirmation. The following grid is fixed before inspecting its results.
All conditions, seeds 0/1/2, and identities are retained. The original paper's
archived comparisons are preserved separately.

Reproduce from the repository root in the project Python environment:

```sh
python -m model.Paper.reviewer_replicated_controls --data /path/to/paper-dataset.json --mode enrollment
python -m model.Paper.reviewer_replicated_controls --data /path/to/paper-dataset.json --mode temporal
python -m model.Paper.reviewer_replicated_controls --data /path/to/paper-dataset.json --mode feature
python -m model.Paper.reviewer_strengthened_report --data /path/to/paper-dataset.json
python -m model.Paper.final_revision_audit --data /path/to/paper-dataset.json
python -m unittest model.test_paper_pipeline model.test_reviewer_controls
```

The summary step reconstructs session membership from the private dataset,
verifies actual training counts and split isolation, checks every aggregate,
and evaluates the fixed enrollment rule. Aggregate results and split ranges
are stored in `reviewer-strengthened/summary.json`; each mode also contains
a complete `results.json`. The private participant dataset and fitted models
are excluded from the publication package.

## Enrollment

Include identities with at least 27 complete sessions before fitting any model.
This permits at least 20 training sessions after the 15%/10% holdouts for every
split. Keep this cohort and each split's validation/test sessions fixed across
budgets. Use nested exact budgets of 1/5/10/20/all sessions per identity and fit
logistic regression, CNN-GRU, and GRU-only. Record actual counts. Each session
contains five movements. Training amounts change across the whole cohort,
including impostor examples; this is not an isolated genuine-only learning curve.

To avoid defining stability after observing results, a usable minimum will be
reported only if every tested split has strict FAR <=1% and FRR <=10%, and
moving to each larger tested budget changes FAR by <=1 percentage point and FRR
by <=5 percentage points in every split for that model. Otherwise report that
no tested budget meets this operational definition. This is a study-specific
criterion, not a universal biometric standard or a deployment recommendation.

## Temporal structure

Use only normalized x/y positions: no elapsed time, inter-event time, button
state, derivatives, path progress, or target-relative channels. Keep all ten
identities and the same 128-step representation. Fit each neural architecture
on ordered, jointly permuted, and reversed samples, for each split. Shapes,
parameter counts and marginal inputs stay fixed within each architecture.
Include logistic regression on the eight distributional summaries (four per
coordinate) as an order-invariant comparator. Joint permutation preserves the
two coordinates at each point; geometry may still convey information about
plausible paths, but explicit time and derivative cues cannot restore ordering.
No mutual-information estimator or causal claim is implied.

## Feature quality

Compare logistic models on full versus low-order channels, using exactly
mean/std/min/max per selected channel in both conditions. Neither condition
receives the extra 13 derived summaries. This separates their removal from the
channel-group removal, unlike the initial reviewer control. Dimensions are
108 versus 76. The fitted coefficient count changes as intended by the ablation.

## Fitting and reporting

Neural settings: up to 50 epochs, patience 8, batch size 32, Adam 0.001. Reset
the seed per condition, enable deterministic TensorFlow operations, and use one
intra/inter-op thread. Fit normalization on training data and calibrate thresholds
on validation data only. Report K=5 macro FAR/FRR/EER and pooled counts, all
three splits, per-user outcomes, and paired differences; distinguish strict
operating-point rates from ROC-derived EER. Overlapping splits are not independent
population replications. The original AI section was preserved during these
fits; on 2026-11-01 the user explicitly requested an updated, consolidated
disclosure covering code, additional analyses, manuscript revision, and the
response letter. That later disclosure does not change experimental records.

Uncertainty qualification: pooling reused attempts across verifiers does not
create independent Bernoulli trials with a common error probability. Binomial
intervals from this evaluation are illustrative under that assumption, not
validated population coverage. The archived bootstrap and permutation
summaries are not participant/session-cluster adjusted. External confirmation
must prespecify dependence-aware inference appropriate to its sampling design.
