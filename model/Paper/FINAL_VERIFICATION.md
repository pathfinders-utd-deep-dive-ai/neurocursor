# Final technical verification — November 1, 2026

## Deliverables

- Clean manuscript: 17 pages, 21 tables, eight figures.
- Point-by-point response: five pages, all eight original reviewer comments,
  plus follow-up reporting clarifications and manuscript page references.
- Marked review copy: 25 pages, compared with the originally supplied main.tex.
  Selected tables are marked as whole blocks to prevent broken diff layouts.
- Editable Overleaf source and complete revision package.

## Checks completed

- Final pass repeated the regression and evidence audits, rechecked primary
  publication records, package checksums, source/PDF equality, response locations,
  and page numbers. Clarified known-impostor scope in the abstract and pooled
  versus macro aggregation in the repeated-results table captions.

- Read both original reviewer reports and checked the corresponding responses.
- All 17 pipeline/control regression tests pass, including every table/figure
  being referenced outside its float and clean run-in heading punctuation.
- Existing evidence audit: 72 configurations, 585 fitted verifiers, 21,060
  confusion-derived metric checks, threshold provenance, session separation,
  nested budgets, ROC/AUC/EER, 64 unique labels, and 21 cited reference keys.
- All 43 original archived evidence hashes match: old evidence was not rewritten.
- Identity-exclusion audit: 270 matched pairs, exclusion from training and
  validation, identical test sessions, no session overlap, 8,640 metric checks,
  and recomputed split means/ranges. Summary preprocessing is training-only.
- Verified 5/17/27 channel and 20/68/121 classical-input counts, CNN output
  32 × 64, target-center semantics, vector jerk, and coefficient group counts.
- Preserved the current AI development-and-writing disclosure byte-for-byte.
- All three PDFs built without unresolved references, duplicate labels, or
  overfull text boxes. Every rendered page was visually inspected; the enlarged
  coefficient figure has readable labels and a corrected IQR caption.
- Sources, figures, PDFs, and archive members are checked for byte identity
  during packaging. Private cursor samples and fitted weights are excluded.

## Remaining submission boundary

This is a completed technical revision, not a certification of scientific
perfection or editorial acceptance. The main study uses known impostors; the
new excluded-identity control is logistic-only and reuses the same cohort.
An isotropic-normalization experiment, external cohort, longitudinal/device
study, and deliberate-attack study were not performed; their limits are stated.

No written institutional OHSP/IRB determination has been supplied. The paper
truthfully describes author-reported verbal guidance and does not claim formal
approval/exemption. Obtain written institutional clarification and all-coauthor
approval before resubmission. No further email contact is authorized.
