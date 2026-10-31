# Phased FlowER reranking implementation plan

**Previous validation-only execution, 2026-10-31:** the authorized fresh nonphysical linear frequency-residual experiment is complete and its all-validation model is frozen. All 21,723 official-validation rows were accounted for; 21,631 supplied an eligible soft-positive fitting loss. Local preparation and training consumed 417.9 CPU-seconds, within the two-CPU-hour cap. The new implementation is `scripts/train_learned_residual.py`; the legacy pairwise trainer was not used. DeepMech, physical calculations, new FlowER inference, cloud execution and reserved-test outcomes remain held. Existing predictions, checkpoint, earlier pilots and fixed test membership stay preserved.

Validation-only artifacts: [learned residual validation experiment](results/reranking/learned-residual-validation-20261031/). Sixteen new fit-level tests and 22 existing phase-contract tests passed before execution. The detailed protocol below and older scope amendments are retained as history; that authorization applied only to local nonphysical validation training and model freezing.

[Development report](results/reranking/learned-residual-validation-20261031/REPORT.md): nested grouped accuracy 85.076% → 85.269%, 691 corrections and 649 new errors, paired gain +0.193 points (95% group-bootstrap interval −0.076 to +0.458). This does not establish a positive effect or meet the provisional one-point goal. [Interpretation and verification](results/reranking/learned-residual-validation-20261031/DEVELOPMENT_REVIEW.md) separates this development result from the still-pending reserved test. All 206 fits converged from zero initialization; all 21,723 saved out-of-fold decisions reproduced after checkpoint reload.

**Learning-curve follow-up completed:** the user authorized separate 25%, 50% and 100% fitting-group copies on the same held-out groups. With five fixed subset seeds and the existing fold-specific settings, all 55 fresh fits converged. Held-out accuracy was 85.236%, 85.278% and 85.269%; doubling 50% → 100% changed accuracy by −0.009 points (95% paired interval −0.138 to +0.123). The current curve is essentially flat. [Results and post-hoc diagnosis](results/reranking/learned-residual-learning-curve-20261031/INTERPRETATION.md) favor examining features/model behavior and label ambiguity before more generation. This is conditional on historical tuning inside each full fitting partition, not a total-label-budget curve. The original frozen model remains unchanged, and all other execution holds persist.

```bash
.venv-analysis/bin/python -m unittest discover -s tests -p test_learned_residual.py
.venv-analysis/bin/python -m unittest discover -s tests -p test_phased_reranking.py
.venv-analysis/bin/python scripts/train_learned_residual.py prepare --out results/reranking/learned-residual-validation-20261031
.venv-analysis/bin/python scripts/train_learned_residual.py train --out results/reranking/learned-residual-validation-20261031
```

Preparation requires a fresh directory. If interrupted before completion, the training command supports `--resume` with verified fit checkpoints and the remaining cumulative CPU budget. A frozen model cannot be refitted in place. The fixed 20,000 test rows still need separately authorized candidate generation before final evaluation.

**2026-10-31 scope amendment:** the next proposed experiment is now the standalone [energy-selection feasibility pilot](ENERGY_FEASIBILITY_PLAN.md), comparing frequency directly with frequency plus energy on matched validation cases. It does not require the nonphysical model below to be trained first. The original phased design remains here as deferred background; the completed Phase 0 output and its frozen protocol are unchanged. The narrower pilot also remains on hold.

## Phase 1 — additive nonphysical residual model

Implement a fresh scorer with `s_j = log(count_j / 64) + lambda * w^T z_j`. The prior coefficient is one. Keep the original frequency winner unless the best alternative exceeds its score by a **strictly greater** tuned margin. Stable original ordering breaks ties. `lambda=0` is an explicit policy, including when all learned policies are worse. Empty pools retain an empty prediction and count as incorrect; unsupported row features trigger frequency fallback. Candidate states/counts/order are immutable.

The old `targeted_reranker.py` uses `(1-alpha)*log_frequency + alpha*learned_score` and a different fitting loss. Do not resume `run_full_validation_experiment.py` or call its legacy `fit` command as this experiment. The new score/labels/folds need a separate trainer and frozen receipt. `phased_reranking_contracts.py` implements the tested score, label, weighting and correspondence contracts; it is **not a fitted model or a complete trainer**.

| Component | Prespecified implementation |
| --- | --- |
| Development data | All 21,723 official-validation rows and their 49,600 saved top-10 candidate entries. The inspected test prefix informs exploratory hypotheses only. |
| Separation | Five outer groups from `validation-identities.json`; inside each outer fitting partition, four inner folds of the same connected groups. Groups connect exact model-condition arrays, normalized inputs and related network components. Network IDs and source row IDs never enter features. |
| Targets | Primary: soft-label cross entropy. Within each fitting partition build a normalized-input label-count catalog, retain positive mass for **every** recorded outcome, and normalize that mass over candidates present in the pool. Report omitted label mass; if no positive is available, omit the row's loss, retain it in all assessment denominators. A candidate without a recorded label is a contrast for supervised prediction, not proven invalid chemistry. |
| Frequency residual | Linear correction without a candidate-common intercept. Initialize new coefficients at zero. Fit L2-regularized soft-label loss with deterministic optimization, record convergence, reject nonconvergent models. Never load prior pilot weights. |
| Features | Inference-only frequency/gaps/diversity; candidate no-change; map-free bond/charge/H inventory differences; local structural fingerprints and reactive-environment descriptors; full-mixture donors, acceptors, ions, solvents when represented; explicit validity warnings. Include all mixture fragments. No training-support/rarity reward. Fixed hashed features need no learned vocabulary; any learned vocabulary/filter/scaling uses fitting data only. |
| Forbidden inputs | Reference-derived terminal flags, target mappings, reference edits, step index, sequence/network IDs, target rank or correctness as predictive features. Labels may only enter the supervised loss, fitting-fold label catalog and permitted training weights. |
| Atom correspondence | Saved candidate strings have **no atom maps**. Start with map-free descriptors. Mapped edits require verified generation provenance or independently validated input/candidate correspondence whose relevant edits are invariant across admissible symmetric mappings. Element conservation or an arbitrary mapper assignment is insufficient. Never recover a candidate mapping through the reference. Record unavailable/ambiguous correspondence. |
| Proton-transfer interpretation | Same heavy bonds alone is an H/charge comparison, not confirmed proton transfer. A verified graph transfer needs correspondence, the relevant H attachment changes and consistent charge/electron changes; this still does not establish a kinetic mechanism. |
| Weighting comparison | Uniform versus bounded failure emphasis: weight two only when a fitting-row pool contains a positive and the frequency winner has zero fitting-catalog positive mass; otherwise weight one. Normalize within the fit partition. Do not emphasize competing positive outcomes. Keep all correct examples, rare successes and valid terminal positives. |
| Initial grid | L2 penalty `{0.01, 0.1, 1, 10}`; `lambda={0, 0.1, 0.25, 0.5, 1}`; margin `{0, 0.1, 0.25, 0.5}`; two weighting policies. Scaler and loss fitting occur in inner fitting folds; choose lambda/margin/weights on inner held-out predictions only. |
| Selection | Maximize all-row strict accuracy on inner out-of-fold predictions; exact ties prefer fewer harms, then fewer interventions, then stronger regularization. No chemical override rule gets automatic priority over zero intervention. |
| Honest development estimate | Score each outer held-out fold once using the policy selected entirely inside its fitting partition. Build no fitting catalogs from its held-out labels. Aggregate all-row paired outcomes. The Phase 0 diagnostics already informed feature design, so these remain development estimates, not a pristine final benchmark. |
| Final refit | After development review, select configuration using grouped validation-only CV, refit preprocessing/catalog/coefficients on all validation, and freeze a separate receipt before accessing test outcomes. Include code/runtime/feature names, all hashes, fallback policy and bootstrap settings. |
| Compute cap proposed for approval | Local CPU, no rental: at most two CPU-hours for the first complete linear experiment, resumable at fit boundaries with fold receipts. This is a proposed cap, not an observed runtime. |

A single modest boosted-tree alternative may be registered **after** the linear baseline is established: shallow trees (maximum depth 3, maximum 200 trees), fixed bounded grid, the same nested groups/soft positive treatment/fallback and frequency residual. No architecture expansion from test feedback. If the library cannot support the prescribed grouped soft-label objective, do not substitute ordinary binary “this row's target versus every other candidate” labels.

## Phase 2 — bounded physical-feature feasibility and ablations

Entry requires an identifiable unresolved chemical distinction in validation, a verified or adjudicated interpretation, enough all-row opportunity to matter, and explicit compute authorization. Quantify affected rows/groups and the maximum incremental gain before spending. The narrow carbon flags and the rarity shortlist alone do not supply a one-point all-row benefit. A nonsignificant linear result is **not** the entry criterion and does not establish that energy features will fail or succeed.

Proposed first cap: at most 100 validation cases, roughly half unresolved recoverable errors and half matched correct-baseline controls from separate groups; include matching mixture size, charge, motif, frequency gap, no-change status and rare successes. Freeze the selection before physical results. Score all retained candidates under one protocol. Preflight the number of unique fragment jobs; cap at 720 new jobs, two workers, 20 wall-seconds per fragment, and four CPU-hours total. Unfinished or excluded jobs become explicit missing features. A two-conformer repeat on a prespecified small subset must fit inside the same total cap. No cloud rental is implied.

Use the existing `score_xtb_candidates.py` as a starting adapter, adding the global budget and audit controls before any run. It currently computes GFN2-xTB energies from independently optimized fragments, uses ETKDGv3 geometries, checks reconstructed topology, and abstains for unresolved radicals/odd electron counts and unsupported cases. It is **not a pKa estimator**.

| Ablation | Required assumptions and records |
| --- | --- |
| Nonphysical only | Frozen Phase 1 policy, evaluated on every feasibility row. |
| Nonphysical + energy | Explicit solvent assumption, conformer protocol/seed, charge and spin per fragment, optimization status, topology preservation, element/isotope/net-charge conservation, units, reference state, cache/binary/code hashes and cost. Preserve candidate topology and strings. Independent-fragment sums omit mixture interactions; do not call their differences barriers, rates or guaranteed preferences. |
| Nonphysical + pKa | Separate estimator or reference thermodynamic cycle. Specify donor/acceptor sites, protonation states, temperature, reference compounds, uncertainty and compatible solvent/standard-state scales. An aqueous pKa cannot silently be combined with a DMSO or gas-phase value. A raw electronic-energy difference is not a calibrated solution pKa. |
| Both | Only after separate energy and pKa ablations; same development folds and fixed primary-method designation before final evaluation. |

The represented solvent molecules do not identify their concentrations or necessarily the bulk solvent. Where a solvent or 298.15 K assumption is used, label it as assumed. Record the actual xTB reference-state option: its documented default has no solution-state correction; the production `bar1mol` choice and any different convention must be explicit. These options are documented in the [official xTB solvation guide](https://xtb-docs.readthedocs.io/en/latest/gbsa.html#reference-states).

**Fallback:** if any required physical score for the compared pool is absent, incompatible, unconverged, changes topology, or has unresolved charge/spin/reference-state meaning, return the frozen nonphysical policy for that row. Do not drop it, replace its structure, rank only the scored subset, or treat score failure as chemical evidence. Add separate coverage/failure tables and paired comparisons on the full feasibility population. Physical calibration, preprocessing, weights and thresholds remain fold-local.

## Phase 3 — one frozen evaluation

Keep the existing 20,000 source-row selection and original generator/checkpoint/64-draw configuration. Final freeze precedes the evaluator's access to reserved targets, including new inference that internally reads labels. No additional sampling for difficult cases or pool edits. New inference and rental budgets require separate authorization. Exact input grouping should be checked from inference-available inputs before unsealing outcomes; if extra links are found, merge them by the frozen rule and update the input-only power check before scoring outcomes.

Freeze one primary method and all secondary ablations. Freeze code, model, preprocessing, fitting label catalog, candidate hashes, fallback, thresholds, all selection manifests, the reference normalization, and the report schema. For secondary inferential claims use a prespecified Holm family or mark intervals descriptive; do not select the reported winner from test outcomes.

Primary paired estimator: `sum(correct_reranked - correct_frequency) / 20000`. Draw **9,999** seeded bootstrap replicates of complete connected input/network groups with replacement; each replicate uses the ratio of total paired changes to total rows, never the unweighted mean of group accuracies. Use the 2.5% and 97.5% quantiles. Retain every row, including empty pools, unavailable positives and all fallbacks.

Report strict top-1 accuracy, paired change/interval, corrections, new errors, wrong-to-wrong switches, intervention rate, fallback counts, group counts and per-method candidate identity checks. Report physical feature coverage, failure causes, CPU/GPU time and observed cost separately.

Prespecify stopping/no-change, verified graph proton-transfer, other structural change, unresolved correspondence, ambiguity and rarity strata. Keep target-derived **reporting** annotations in a separate evaluation-only object created after freeze; never feed them into the score. If a reference-based stopping or ambiguity stratum is reported, label it as outcome-side diagnosis. Same heavy-bond structure without verified H/charge correspondence belongs to the unresolved category. Report overlapping strata explicitly and give their denominators/group counts. Rare/common refers to frozen training-support definitions used for reporting, not a scoring bonus.

Set-valued agreement is secondary and diagnostic: report separately (a) agreement/coverage against the frozen fitting catalog and (b) test-internal recorded-label agreement after unsealing. The latter includes evaluation labels and is not evidence that the model inferred the exact target or experimental chemical validity. Unknown catalog inputs must not be silently treated as known invalid outputs.

Decision language: a lower paired CI above zero supports a positive effect; a point estimate at or above one point meets the provisional target only at the point-estimate level; a lower CI above one point supports a gain exceeding that target. If the upper CI is below one point, the worthwhile gain is ruled out at that interval level. Otherwise a nonsignificant result is inconclusive. No retuning, subgroup-driven threshold changes, or test-informed refitting.

## Working commands and remaining implementation

The completed code executes Phase 0 and tests the later scoring contracts. The learned trainer, mapping validator, physical extension/pKa adapter and frozen assessment runner are still implementation work; no training command is claimed ready. The legacy trainer must not be used as a substitute for this protocol.

From `/Users/deveshi/Desktop/Flow-ER`, reproduce Phase 0 into a **fresh** directory:

```bash
.venv-analysis/bin/python scripts/phase0_reranking_audit.py --out results/reranking/phased-reproduction-NEW
.venv-analysis/bin/python scripts/phase0_group_power.py --audit results/reranking/phased-reproduction-NEW --workers 4
.venv-analysis/bin/python scripts/write_phase0_report.py --audit results/reranking/phased-reproduction-NEW
```

The audit refuses to overwrite an existing output directory. Use a new suffix for a second reproduction; the original frozen selection is reused and never resampled. `--workers 4` runs CPU power simulations, not training or GPU inference.

```bash
.venv-analysis/bin/python -m unittest discover -s tests -p test_phased_reranking.py
.venv-analysis/bin/python scripts/phased_reranking_gate.py --experiment results/reranking/phased-20261031 --phase 1 --check
.venv-analysis/bin/python scripts/phased_reranking_gate.py --experiment results/reranking/phased-20261031 --phase 2 --check
.venv-analysis/bin/python scripts/phased_reranking_gate.py --experiment results/reranking/phased-20261031 --phase 3 --check
```

The gate has no launch capability. An attempted `--execute` exits with the user's hold reason before loading candidates or writing files. Opening a gate later requires explicit user authorization and the corresponding implemented/reviewed runner; a CLI flag alone is not authorization.

Meaningful tests already implemented: exact versus normalized identity, transitive grouping, tied-majority counting, empirical ceiling arithmetic, held-out-label independence of lookup, unavailable-majority fallback, leakage rejection, reserved-target read rejection, execution hold/no-write behavior, correct row-weighted paired bootstrap and compressed-group equivalence, synthetic paired-count conservation, additive residual scoring, strict margins, zero policy, unsupported-score fallback, soft positive labels, bounded weights, and correspondence/provenance rejection. Before a trainer is authorized, add fit-level tests showing that changing outer held-out labels cannot change preprocessing, features, fit catalogs, weights, coefficients or chosen thresholds; gradient checks; convergence and checkpoint/resume identity; candidate-pool immutability; physical failure fallback and reference-state cache separation; and frozen evaluation refusing mismatched hashes.
