# FlowER energy-selection feasibility experiment

## Material passport

academic-research-suite / experiment-agent / plan; 2026-10-31. Status: **prepared, not run**. This amendment adopts the user's narrower question. The completed Phase 0 audit and historical phased protocol remain preserved. It proposes an independent energy pilot; a fitted nonphysical reranker is not a prerequisite.

**Question:** Does adding an approximate energy score correct chemical selection errors more often than it damages correct predictions?

**Execution holds remain:** no physical calculations, parameter tuning/model training, new inference, cloud rental or reserved test-outcome access. The scoring contract is implemented and can be tested using synthetic numbers. Cohort selection, adjudication, physical-scoring integration and the assessment runner are not complete.

## Cohort and separation

Use only the existing 21,723 official-validation cases and their unchanged saved candidate pools. The inspected test prefix can motivate hypotheses, but supplies no pilot observations. Leave the fixed 20,000-row test selection and its outcomes untouched.

Propose at most **50 matched pairs / 100 rows**: one interpretable, top-10-recoverable baseline error and one baseline-correct control per pair. Focus on proton-acceptor/site choice and justified continuation versus stopping; do not select on rarity or computed energy. Exclude a witnessed competing recorded winner from the primary chemical-error cohort. Absence from the catalog is not itself chemical invalidity.

Use the existing transitive exact-input/normalized-input/reaction-network groups. Select at most one row per group. Partition groups deterministically before matching or energy calculations, using seed 20261031; aim for **30 development pairs and 20 assessment pairs**. Matching occurs within each partition, with different groups on the two sides. Match chemical motif, mixture size, charge, candidate diversity, frequency gap and no-change status where feasible. Include rare correct predictions and valid stopping controls. Document unmatched candidates and any compromises. Freeze membership, pairing, candidate hashes and the actual group counts before computing energies. If too few defensible pairs exist, reduce and document the sample before calculation; do not fill with ambiguous cases just to reach 100.

Chemical review must examine the complete mixture and candidate set without energy scores. Record whether the proposed distinction is defensible, unresolved, or a recorded alternative, and who made the judgment. The reference is not automatically the chemically preferred next state. Saved candidate strings have no atom maps: verified proton-transfer/site-edit annotations require independently checked correspondence; do not obtain it from reference mappings. Same heavy-atom bonds alone do not establish proton transfer. Unresolved chemical cases cannot support a claim of chemical correction, even if they support strict-label agreement.

Case/control construction intentionally uses baseline correctness and pre-existing diagnostic catalogs. It is a selected validation feasibility experiment, not a pristine test. Selection catalogs and annotation provenance are retained separately; neither supplies predictive features. Any fitting catalog must use development groups only. Assessment labels are withheld from parameter selection after cohort construction.

## Two policies only

Baseline: original sampling-frequency order. Proposed intervention:

```
delta_E(c) = E(c) - E(original_frequency_winner)     # kcal/mol, within one row
S(c) = log(count(c) / 64) - lambda * delta_E(c)
```

The energy reference is the **frequency winner**, never the recorded target. Lambda is nonnegative, in mol/kcal; the switching margin is dimensionless. Switch only when the best alternative exceeds the frequency winner's score by strictly more than the margin. Preserve original order for ties. Include `lambda=0` as the baseline policy. No rarity bonus, pKa feature, learned structural correction, or energy-only arm is included in this pilot.

Proposed development grid: `lambda={0, 0.01, 0.03, 0.1, 0.3, 1}` and `margin={0, 0.1, 0.25, 0.5, 1}`. Select by net strict corrections on all development rows, including fallbacks; ties prefer fewer new errors, then fewer interventions, then smaller lambda and larger margin. Collapse equivalent zero-intervention policies to baseline. Freeze the chosen parameters and all physical rules before scoring assessment labels. Do not retune from assessment results. No scaler or calibration is fitted using assessment energies or labels.

**Score every candidate in each frozen saved pool** (the retained top-10 entries), including no-change candidates, and every fragment with its multiplicity. Do not score only the reference/winner, truncate the saved pool, insert candidates, optimize the target preferentially or resample FlowER. Below-top-10 candidates are outside this frozen comparison.

## Physical protocol and fallback

Proposed method: GFN2-xTB with ALPB solvation, using the existing isolated-fragment scorer as a starting point. Every candidate in a row uses the same declared solvent, reference state and geometry protocol. Declare each row's solvent assignment and evidence **before energies**. Molecular presence alone does not establish the bulk solvent. Missing, conflicting or insufficient solvent context triggers frequency fallback; do not silently use gas phase or guess water. If this leaves few eligible cases, that is a feasibility limitation to report.

Use an explicit `bar1mol` reference-state convention, recording the exact installed-version argument and emitted settings. The official [solvation guide](https://xtb-docs.readthedocs.io/en/latest/gbsa.html#reference-states) and [command-line page](https://xtb-docs.readthedocs.io/en/latest/commandline.html) describe defaults differently, so relying on an implicit default is unacceptable. Verify the installed version's syntax before any authorized calculation. xTB's electronic temperature must also be recorded explicitly; the proposed 300 K electronic setting is not a measurement of reaction temperature.

Generate two ETKDGv3 conformers per fragment with seed 42 and a single thread; use the minimum score after the prescribed optimizations. Record attempted/generated/converged conformers and duplicates. Require both requested conformers to be generated, converged and topology-preserving; otherwise mark that fragment unreliable. This conservative rule is frozen before results. Use formal fragment charges; closed-shell supported fragments only. Unresolved spin/radicals, isotopes unsupported by the XYZ pipeline, unsupported elements, or fragment size beyond the frozen cap trigger failure. Check full-mixture element/isotope inventory and net charge, as well as fragment topology, bond orders, H attachment and charges after optimization. Do not replace a candidate with the structure it relaxes into.

The feature is the sum of solvation-corrected xTB total energies of independently optimized fragments, converted to kcal/mol. It lacks a full solution free-energy treatment, intermolecular interactions and activation barriers. Fragment-number changes and missing concentration/entropy effects require explicit reporting. Lowest next-state energy need not identify the kinetically preferred next step. This experiment estimates whether this particular approximation is useful, not whether energy in general determines the answer.

**Whole-row fallback:** uncertain solvent, incompatible protocol/reference states, failed/unreliable calculation or changed structure in **any retained candidate** returns the original frequency prediction for the entire row. Missing scores never promote another candidate. Keep every selected row in assessment denominators and report failure reasons and complete-pool coverage. Do not rank only the successfully scored candidates or convert a failure flag into a prediction feature.

Proposed local CPU budget: at most 720 new unique fragment jobs, two workers, one thread per worker, 20 wall-seconds for each fragment job including both conformers, and two hours elapsed time (at most four configured worker-hours). Preflight the entire selected cohort, including all candidates. If its job estimate exceeds the cap, revise and re-freeze the design before energies, not after observing outcomes. A reached budget marks all unfinished work missing and invokes fallback; no automatic retries. Record actual CPU time as well as elapsed time and counts. No GPU or cloud rental is proposed.

## Assessment and interpretation

On the separate assessment groups, report raw counts and rates of **corrections, new errors, wrong-to-wrong switches, unchanged predictions, interventions and fallback**, plus strict top-1 accuracy and paired change. Separately report chemically adjudicated improvements/harms; label-only switches must not be called confirmed chemical corrections. Retain and disclose unresolved judgments.

Report the correction rate among initially wrong cases and the damage rate among initially correct controls. The balanced sample's net count does not estimate population accuracy. In a comparable population with baseline-error prevalence `p`, projected net change would be `p * correction_rate - (1-p) * damage_rate`; this projection requires representative rates and independently justified prevalence. Do not extrapolate it to all validation or test rows from this selected sample.

For this paired sampling design, bootstrap complete matched pairs (9,999 resamples; seed 20261031), retaining both rows and their original group IDs. No source group may occur in multiple pairs or partitions. Compute ratio-of-total paired changes to total rows; also report rescue/damage intervals and individual counts. With only about 20 assessment pairs, intervals may be broad. The previous 20,000-row power simulation does not establish power for this enriched pilot.

A positive paired interval supports an effect in this selected cohort. Corrections exceeding harms only at the point-estimate level is a preliminary signal. An interval spanning benefit and harm is inconclusive. The earlier +1-point full-benchmark target remains a later population-scale criterion; it is not a feasibility-sample success threshold. Report coverage, conformer sensitivity limitations and cost alongside any effect. Further physical work or an overall-accuracy claim requires a separately authorized, representative experiment. No automatic expansion or test retuning.

## Implementation and safe commands

`scripts/energy_selection_contract.py` implements only the pure scoring/fallback decision. The legacy `score_xtb_candidates.py` defaults to K=5, has a different conformer-failure policy, lacks explicit reference-state control and a global budget, and must not be launched unchanged for this protocol. Before an authorized run, implement those controls, candidate-hash/protocol checks, a frozen cohort/pair manifest and a label-separated tuning/assessment runner. Test label independence, group separation, complete candidate accounting, cache separation by solvent/reference state, failed-fragment fallback and budget interruption using mocks.

These commands perform synthetic tests and inspect existing holds only:

```bash
.venv-analysis/bin/python -m unittest discover -s tests -p test_energy_selection_contract.py
.venv-analysis/bin/python scripts/phased_reranking_gate.py --experiment results/reranking/energy-feasibility-20261031 --phase 2 --check
```

There is no ready or authorized physical-run command. The saved machine protocol records the same holds. Existing FlowER weights, saved pools, Phase 0 results and prior models remain unchanged.
