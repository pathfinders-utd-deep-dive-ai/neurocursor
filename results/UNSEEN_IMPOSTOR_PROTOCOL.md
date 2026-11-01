# Known versus excluded-impostor identity control

This exploratory control is specified before fitting its models. It reuses
the existing private dataset; it is not a new population or attack study.

- Model: the existing fixed full-feature logistic regression (121 summaries,
  L2, C=1, lbfgs, balanced class weights, max_iter=2000).
- Use the existing session splits with seeds 0, 1, and 2; K=5 only.
- For every claimed identity u and different held-out identity h, evaluate
  the identical genuine-u and impostor-h test sessions in both conditions.
- Seen condition: fit on all training identities and calibrate using the
  original validation identities. Fit one model per claimed user and split.
- Excluded condition: remove every h sample from training and validation,
  including preprocessing estimation. Genuine-u training/validation sessions
  are unchanged. Fit a model for each (u,h) pair.
- Fit summary scaling only on each condition's permitted training rows.
  Calibrate both validation-EER and strict-zero-validation-FAR thresholds on
  permitted validation attempts. Never use h to calibrate the excluded fit.
- There are 90 ordered u/h pairs per split, 270 comparisons per condition,
  30 seen model fits, and 270 excluded model fits. Retain all outcomes.
- Report equal-pair-weight macro FAR/FRR/EER/AUC for each seed and their means
  and split ranges. Counts, scores, source indices, and calibration records
  are archived. Reused genuine attempts and overlapping splits are dependent;
  do not treat the pair count as independent population trials.
- This isolates exclusion of one observed impostor identity from fitting and
  calibration. It does not establish external-user generalization, unseen
  devices, deliberate attack resistance, or neural-model performance.
