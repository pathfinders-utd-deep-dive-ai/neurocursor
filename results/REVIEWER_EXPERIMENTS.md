# Initial additional reviewer experiments (archived)

The manuscript now uses the stronger replicated controls described in
`REPLICATED_REVIEWER_PROTOCOL.md`. This file documents the initial revision
experiments, preserved for traceability; its capped budgets and one-split
neural controls should not be confused with the strengthened results.

These retrospective development experiments address Reviewer 1 comment 3 and
Reviewer 2 questions about enrollment scale, feature quality, and temporal order.
They supplement, rather than replace, the archived July comparisons.

Processed dataset SHA-256:
`ef5df46c0f3c02bf458084e435079c4911026d81cea36f391823ec63f0570a95`.
This exactly matches `paper-real-data-2026-07-29/SENSITIVITY.json`.
The private data are not included in this repository.

Run from the repository root in the project Python environment:

```sh
python -m model.Paper.reviewer_experiments --data /path/to/paper-dataset.json --seeds 0 1 2
python -m model.Paper.reviewer_experiments --data /path/to/paper-dataset.json --output results/reviewer-neural-controls --seeds 42 --neural
```

`reviewer-experiments/results.json` contains all 21 logistic conditions.
`reviewer-neural-controls/results.json` contains seven seed-42 logistic conditions
and six additional neural conditions. Each record includes the selected movement
indexes, fixed validation/test indexes, actual per-user session counts, dataset
hash, user metrics, pooled counts, and configuration/runtime metadata.

Training sessions are ordered once per user and split, then nested caps of
1, 3, 5, 10, 20, and all sessions are used. Validation/test sessions are unchanged
within a split. A cap is truncated when a user has less history. This retains
all ten identities; it does not assert exact enrollment at the nominal cap.
Both genuine and impostor training data are affected by the cohort-wide caps.
One session contains five movements. No test-driven stopping rule is used.
The reported learning curve is logistic, not a neural learning curve.

Neural conditions reset seed 42, use 50 maximum epochs, patience 8, batch size 32,
learning rate 0.001, and the existing CNN-GRU and GRU-only implementations.
Permutation jointly reorders the 128 resampled vectors within each movement
in every partition, preserving exact channel marginals and same-sample channel
relationships. This tests sensitivity to chronological order, not mutual
information. It does not remove temporal information already encoded in explicit
time or derivative channels. Fresh neural fits differ from historical fits;
results are reported separately and no untouched-test claim is made.

Low-order inputs remove acceleration (three channels), jerk (three), angular
velocity, and curvature, leaving 19 channels. Classical low-order inputs contain
76 distribution summaries; the existing summary routine also omits all 13
derived statistics when its required channels are absent. Consequently the
classical contrast does not isolate a single noisy feature.

Three overlapping logistic splits and one neural split do not establish a
population enrollment minimum or a causal architecture explanation. The
manuscript reports all outcomes and retains these limitations.
