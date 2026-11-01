# Reviewer request traceability

| Request | Revision and evidence |
| --- | --- |
| R1.1: Remove overlap between Introduction and Related Work | Both sections rewritten; Related Work compares task, inputs, strengths, and limitations in the positioning table. |
| R1.2: State exact inputs and classical summary statistics | Temporal channel table plus exact input table: 5/17/27 primary channels, 19-channel added control; classical dimensions 20/68/121/76. Mean, standard deviation, minimum, maximum and all 13 extra summaries explicitly identified. |
| R1.3: Explain model/feature behavior using more experiments | Strengthened grid: 45 enrollment, 6 corrected classical feature-quality, and 21 timing-free temporal configurations. All three split seeds and all conditions retained. |
| R2.1: Shorten Introduction and clarify contributions | Compact narrative with four numbered contributions and no introductory equations. |
| R2.2: Consolidate Related Work and explain recent work | Single comparative narrative and positioning table; explicit LT-AMouse and device-confounding comparisons. |
| R2.3: Training scale and minimum stable enrollment | Fixed seven-identity eligible cohort; exact nested 1/5/10/20/all-session budgets for logistic, CNN-GRU, and GRU-only, across three splits. Holdouts fixed. An explicit study-specific error/stability criterion is evaluated; no universal population minimum is implied. |
| R2.4: Justify retaining CNN-GRU after rejected H1 | No recommendation to select hybrid. It is documented as the tested candidate and a negative result; GRU-only/classical comparisons retained. |
| R2.5: Temporal information versus acceleration-like features | Normalized x/y only; no explicit time, derivatives, progress, button or target channels. Ordered/permuted/reversed fits for both networks across three splits plus order-invariant logistic comparator. Marginals and model parameter counts held fixed within architecture. No measured-mutual-information claim. |
| R2 overall: Major revision | Manuscript, response, experiment runner, protocol, full aggregate records, feature accounting and figure updated. |

Strengthened protocol: `results/REPLICATED_REVIEWER_PROTOCOL.md`.
Validated aggregates: `results/reviewer-strengthened/summary.json` and
the enrollment/temporal/feature `results.json` files below that directory.
Initial reviewer runs remain archived separately in
`results/reviewer-experiments` and `results/reviewer-neural-controls`.

All 72 configurations passed independent reconstruction of session counts and split
isolation from the source data, metric aggregation, exact nested budgets,
dataset fingerprint, permutation/reversal invariants, and twelve regression
tests. Citations/references resolve, and all 18 manuscript pages and four
response pages were rendered and visually inspected. The AI-Assisted
Development section is byte-identical to its pre-change version.

Scope: the strengthened learning curves and temporal controls cover three
overlapping splits. Enrollment covers seven eligible identities and temporal
controls all ten. These do not establish population stability or causality.
Direct GitHub Codespaces access needs a token with `codespace` scope; the local
checkout and remote `shaurya` branch are verified separately.
