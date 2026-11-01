# Reviewer request traceability

| Request | Revision and evidence |
| --- | --- |
| R1.1: Remove overlap between Introduction and Related Work | Both sections rewritten; Related Work compares task, inputs, strengths, and limitations in the positioning table. |
| R1.2: State exact inputs and classical summary statistics | Temporal channel table plus exact input table: 5/17/27 primary channels, 19-channel added control; classical dimensions 20/68/121/76. Mean, standard deviation, minimum, maximum and all 13 extra summaries explicitly identified. |
| R1.3: Explain model/feature behavior using more experiments | Strengthened grid: 45 enrollment, 6 corrected classical feature-quality, and 21 coordinate-only temporal configurations. All three split seeds and all conditions retained. |
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
dataset fingerprint, permutation/reversal invariants, and sixteen regression
tests. Citations/references resolve, and all 15 manuscript pages and four
response pages were rendered and visually inspected. The AI-Assisted
Development section is byte-identical to its pre-change version.

Final wording audit also aligned the discrete EER definition with the code,
identified canvas-normalized derivative units and training-only scaling,
documented the non-conservative resampled summary estimators, and qualified
test isolation per fit rather than claiming untouched confirmatory evaluation.
Coordinate-only controls are not described as eliminating all implicit timing
or acquisition information. All 60 labels and 16 cited bibliography keys resolve.

The subsequent audit corrected all three additional findings: the pooled
binomial FAR bound is explicitly illustrative under independence and common
error probability, not validated population coverage; pause/click summaries
now have exact operational definitions and regression tests (including the
fractional button-state and missing-click cases); and the research-question
subsection now has a substantive summary rather than a floating table alone.
The report generator preserves the same dependence caveat. Bootstrap and
permutation summaries are disclosed as not participant-cluster adjusted.

Repeatable read-only evidence audit (private dataset required):

```sh
python -m model.Paper.final_revision_audit --data /path/to/paper-dataset.json
python -m unittest model.test_paper_pipeline model.test_reviewer_controls
```

The audit checks 72 configurations, 585 verifiers, 21,060 confusion-derived
metric identities, ROC/AUC/EER consistency, threshold provenance, source session
membership, nested exact training budgets, all 43 archived evidence hashes,
document references, and the protected disclosure hash. It does not rerun
neural fitting or certify participant provenance, consent, or population validity.

Final concision pass: the previous 19-page revision is reduced to 15 pages,
with approximately 27% fewer visible-text words. During that pass, Introduction source-word count
falls from 469 to 332; Related Work from 600 to 472 (including its unchanged
comparison table). All 19 retained table environments were unchanged during
that pass. The removed table only repeated future-validation
requirements now summarized in prose. Redundant feature/evaluation explanations
and overlapping deployment sections are consolidated; three generic neural
representation equations are replaced by exact implemented baseline details.
Experimental input definitions, all empirical tables, critical uncertainty
caveats, reviewer-requested controls, and the protected AI disclosure remain.

Final copyedit removes author-supplied periods from both IEEE run-in headings,
preventing the template's automatic colon from producing `.:`. A regression
test now checks every run-in heading. Terminology distinguishes speed from
vector velocity, ROC-derived EER from calibrated operating-point rates, and
the randomized training-session order from statistical independence. The
low-order input-table row is clarified as a classical summary control, not an
additional neural experiment. No performance-table values have changed.
All eight primary model types and the canonical implementation remain included;
the publication package intentionally excludes trained weights and private data.

Scope: the strengthened learning curves and temporal controls cover three
overlapping splits. Enrollment covers seven eligible identities and temporal
controls all ten. These do not establish population stability or causality.
Direct GitHub Codespaces access needs a token with `codespace` scope; the local
checkout and remote `shaurya` branch are verified separately.
