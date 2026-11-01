# Reviewer and resubmission traceability

## Latest pre-submission audit (supersedes older status below)

- Added matched identity-exclusion logistic control: 270 pairs, three splits,
  identical test sessions, no held-identity training/validation/preprocessing.
  Strict FAR 0.99% known versus 4.07% excluded; Table XVII.
- Reported archived per-verifier permutation p, Cohen d, bootstrap AUC intervals,
  and exact FAR/FRR intervals in Table XXI, with dependence limitations.
- Added fixed classical hyperparameters and no-search disclosure; clarified
  threshold policies, test-ROC EER, macro versus pooled aggregation, and the
  distinct coefficient-group taxonomy including angular velocity.
- Corrected H2 endpoint interpretation and H3 FRR tradeoff; added weak-baseline
  limits to coordinate-order controls and anisotropic-normalization limitations.
- Added speed magnitude to the channel prose; verified vector jerk equation and
  target-center semantics against the acquisition code. Simplified EER brackets.
- Removed revision-meta wording; neutral experiment/check entry points are named
  in the manuscript. Figure 5 caption matches plotted quantities; Figure 7 uses
  readable labels. Standardized random movement-level terminology.
- Clarified Antal/Egyed-Zsigmond's configuration-specific versus best AUC and
  LT-AMouse's segment-length strengths; replaced glossary support with NIST
  SP 800-63B-4. Existing published references and AI disclosure retained.
- Added author-confirmed July 1–31, 2026 dates, four-author collection, online/
  UTD ECSS 2.415 recruitment, informed consent, no funding and no conflicts.
- Removed unsupported legal-exemption claim. Written OHSP/IRB clarification is
  still outstanding: see SUBMISSION_STATUS.md. No further emails authorized.
- All eight original reviewer comments retain individual responses, now with
  printed page references; the two Introduction run-in labels are acknowledged.

The checks below describe prior revisions. Final build counts, test results,
and package verification are recorded separately in FINAL_VERIFICATION.md.

## Original reviewer requests

| Request | Current revision and location |
| --- | --- |
| R1.1: Introduction/Related Work overlap | Separate concise Introduction and comparative Related Work; Sections I--II, Table I. |
| R1.2: Actual neural/classical inputs | Exact channel groups, dimensions, four summary statistics, 13 additional statistics, operational estimators, and architecture; Section V, Tables II--IV. |
| R1.3: Additional data/feature experiments | 45 enrollment, 21 temporal, and 6 corrected feature-quality configurations; Sections VI-G and VII-E--F, Tables XII--XIV. |
| R2.1: Long Introduction/unclear contributions | Compact narrative and four numbered contributions; Section I. |
| R2.2: Fragmented Related Work/latest strengths and limits | Comparative prose and ten-row positioning table; four added 2024--2026 papers plus restored published 2022 BeCAPTCHA-Mouse; Section II, Table I. |
| R2.3: Training scale/minimum enrollment | Three model families, three split seeds, seven eligible identities, exact nested 1/5/10/20/all-session budgets and fixed holdouts; Sections VI-G/VII-F, Table XIII. No tested budget meets the stated stability criterion. |
| R2.4: Justification for retaining CNN--GRU | Retained as a tested negative result, not the recommended architecture; Sections V, VII-B, VIII-A, IX, Table VIII. |
| R2.5: Mutual information versus kinematics | Coordinate-only ordered/permuted/reversed fits across three splits with invariant comparator; Sections VI-G/VII-F, Table XIV. No measured-mutual-information claim. |

The response letter gives a section/table location for each of these eight
points. Its redundant overall-major-revision and internal verification blocks
have been removed; the opening and point-by-point responses cover the revision.

## Follow-up audit corrections

- Every one of 19 tables and eight figures now has an in-text reference outside
  its floating environment. A regression test enforces this.
- Table XV identifies seed 42, full-feature CNN--GRU, validation-EER calibration,
  and individual-movement scores without K-movement aggregation. It is not the
  first-movement-only K=1 attempt protocol.
- Section VI-K links the verified public repository's shaurya branch and names
  the canonical implementation, runners, protocols, and result records. The
  package does not redistribute the private processed dataset or fitted weights.
- The feature-quality control explicitly covers session-split seeds 0, 1, and 2.
- The alarming provenance sentence is replaced with the concrete limitation
  that recruitment is not documented in the analysis file. Author collection,
  supervision, recruitment, and sharing permissions await author confirmation;
  no stronger collection claim has been invented.
- At the user's explicit latest request, the previously preserved AI section
  is now one consolidated development-and-writing disclosure, including
  additional analyses, manuscript revision, and reviewer-response assistance.
  The duplicate Section VI-J paragraph and unverified historical model-version
  claim are removed. The audit records the new disclosure hash.
- Heading punctuation is checked automatically. Speed/velocity terminology,
  ROC-derived EER, and random ordering versus independence were clarified.
- The abstract and conclusion lead with the measured five-movement protocol,
  approximately 4.43 s mean completion time, and 98.78% mean ROC--AUC. High FRR
  and other necessary limitations remain explicit without claiming deployment
  readiness.

## Literature verification

Added primary-source references: Dave et al. (2024), DOI
10.5121/ijcsit.2024.16103; Jin and Liao (2025), DOI
10.1007/s00530-025-01897-0; Tao et al. (2026), DOI
10.1016/j.comnet.2026.112502; Sasikumar and Nagarajan (2026), DOI
10.3389/frai.2026.1914353. Restored Acien et al. using the published 2022
Pattern Recognition article, DOI 10.1016/j.patcog.2022.108643.
LT-AMouse is cited as arXiv:2504.21415v2; the checked record lists no journal
reference, and a separate journal publication was not confirmed.

## Verification and boundaries

The read-only audit checks 72 configurations, 585 verifiers, 21,060
confusion-derived metric identities, ROC/AUC/EER consistency, threshold
provenance, session isolation, nested exact budgets, all 43 original archived
evidence hashes, 61 labels, 21 cited bibliography keys, and disclosure content.
All 17 regression tests pass. Numerical result records and performance-table
values are unchanged.

The clean manuscript is 16 pages and response four pages. The modest increase
from the prior 15-page manuscript supplies the requested recent comparisons,
references, cross-references, and availability statement; no page target is
imposed. A 23-page marked review copy compares the revised source with the
original supplied main.tex. Its legend explains additions/deletions and the
whole-block rather than granular equation comparison; the new positioning
table is likewise marked as an added block. Clean files are the
authoritative submission text. All clean/response/marked pages are rendered
for visual inspection, with clean logs and no undefined references or overflow.

These checks validate the saved evidence and documents, not participant
provenance, consent, a population enrollment minimum, or mutual information.
See AUTHOR_CONFIRMATIONS.md for the remaining author-side factual confirmations
and the journal-specific marked-file requirement that could not be verified.

Recheck from the repository root:

~~~sh
python -m model.Paper.final_revision_audit --data /path/to/paper-dataset.json
python -m unittest model.test_paper_pipeline model.test_reviewer_controls
~~~
