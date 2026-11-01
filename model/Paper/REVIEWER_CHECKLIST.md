# Reviewer request traceability

| Request | Revision and evidence |
| --- | --- |
| R1.1: Remove overlap between Introduction and Related Work | Both sections rewritten; Related Work compares task, inputs, strengths, and limitations in the positioning table. |
| R1.2: State exact inputs and classical summary statistics | Temporal channel table plus exact input table: 5/17/27 primary channels, 19-channel added control; classical dimensions 20/68/121/76. Mean, standard deviation, minimum, maximum and all 13 extra summaries explicitly identified. |
| R1.3: Explain model/feature behavior using more experiments | Added three-split nested logistic learning curve, low-order logistic control, and six fresh neural fits. All conditions retained. |
| R2.1: Shorten Introduction and clarify contributions | Compact narrative with four numbered contributions and no introductory equations. |
| R2.2: Consolidate Related Work and explain recent work | Single comparative narrative and positioning table; explicit LT-AMouse and device-confounding comparisons. |
| R2.3: Training scale and minimum stable enrollment | Measured learning curve with fixed holdouts, caps 1/3/5/10/20/all, and actual per-user counts. Results show scale sensitivity and no demonstrated universal stable minimum. |
| R2.4: Justify retaining CNN-GRU after rejected H1 | No recommendation to select hybrid. It is documented as the tested candidate and a negative result; GRU-only/classical comparisons retained. |
| R2.5: Temporal information versus acceleration-like features | Fresh ordered/permuted/low-order fits for both networks. Permutation preserves exact resampled channel marginals and same-sample relationships. Outcomes are architecture-dependent; no measured mutual-information claim. |
| R2 overall: Major revision | Manuscript, response, experiment runner, protocol, full aggregate records, feature accounting and figure updated. |

Experiment records and protocol: `results/REVIEWER_EXPERIMENTS.md`,
`results/reviewer-experiments/results.json`, and
`results/reviewer-neural-controls/results.json`.

Verification: all 34 recorded conditions checked for metric aggregation,
fixed holdouts and nested budgets; dataset digest matches archived evaluation;
permutation invariants checked; existing eight regression tests pass;
citations/references resolve; final PDFs rendered and visually inspected.

Scope: the logistic learning curve has three overlapping splits, neural
controls one split. These answer the reviewer questions empirically at the
available cohort scale and do not establish population stability or causality.
Direct GitHub Codespaces access needs a token with `codespace` scope; the local
checkout and remote `shaurya` branch are verified separately.
