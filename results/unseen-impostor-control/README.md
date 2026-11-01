# Matched identity-exclusion control

This three-split logistic-only experiment evaluates 270 ordered genuine/impostor
identity pairs under matched test sessions. It is separate from the 72-condition
enrollment/feature/order grid. The complete protocol is in
[UNSEEN_IMPOSTOR_PROTOCOL.md](../UNSEEN_IMPOSTOR_PROTOCOL.md).

| Strict operating point / test ranking | Known identity | Excluded identity |
| --- | --- | --- |
| FAR | 0.99% (0.00–2.96%) | 4.07% (1.85–5.74%) |
| FRR | 32.78% (21.67–45.00%) | 33.77% (27.04–43.15%) |
| Test-ROC EER | 1.30% (0.00–2.22%) | 4.60% (3.52–6.02%) |
| ROC-AUC | 99.05% (98.40–100.00%) | 95.86% (94.44–96.73%) |

Entries are equal-pair-weight means with ranges across split means, not
independent-population confidence intervals. Held identities are removed from
training, summary scaling, and validation in the excluded condition. The main
paper's original evaluation uses known impostor identities; its FAR cannot be
interpreted as FAR for previously unseen people. This exploratory same-cohort
control is not external or deliberate-attack validation.

Run `python -m model.Paper.unseen_impostor_control --data /path/to/private.json`.
Audit records with `python -m model.Paper.audit_unseen_impostor --data /path/to/private.json`.
`results.json` stores split indices, attempt scores/labels, calibration, and
metrics; `summary.json` stores all split means, ranges, and aggregate means.
Private cursor samples and fitted weights are not redistributed.
