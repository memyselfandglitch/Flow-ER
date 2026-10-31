# Grouped reranking analysis after inference

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Date: 2026-10-27
- Stage: label ambiguity audit and 50-case stratified chemical review
- Status: label audit and stratified review complete; full benchmark preserved and paused

## Revised scope (2026-10-28)

The user replaced the previous finish-inference-first plan:

1. Preserve checkpoint, predictions, logs, configuration, then pause inference.
2. Use existing cases diagnostically; initially fit/tune rerankers only on official
   training/validation data. The subsequent request explicitly permits a test
   subset for failure learning; that development must remain labeled exploratory.
3. Compare frequency and frozen reranking on a fixed untouched test subset,
   including baseline-correct and top-K-missing cases. Count rescues and harms.
4. Resume the full benchmark only after credible pilot benefit or when the
   final reproduction result is needed.

Preservation is complete: `results/paused-20261028/` has 57 checksum-verified
files. The H100 is stopped with its preserved backup verified. The prefix has 20,152 rows,
each with 64 samples; 198,845 benchmark rows remain.

**The label audit and stratified review use cached predictions.** In the current inference, top K
means K distinct next-state candidates ranked by sampling frequency for a supplied
elementary-step input. It is not K consecutive correct steps and not K complete
pathways rolled out from initial reactants. The failure-learning pilot was followed
by the user's request to examine whether competing labels explain the apparent
ranking errors. Both studies used cached predictions locally.

## Completed stratified review (2026-10-29)

See the [50-case report](results/stratified-review-50/REPORT.md),
[illustrated case browser](results/stratified-review-50/review.html), and
[frozen selection](results/stratified-review-50/selection.json).

- Fifty cases from fifty networks were selected from the 435 unmatched errors:
  24 connectivity differences, 16 H/charge differences, nine other
  bond-order/H/charge differences and the one empty candidate set.
- Thirty-one references are within top 10, two are rank 11 and 17 are absent
  from all saved samples. Deliberate stratification means these rates must not
  be extrapolated to the population.
- Eleven reviews identify specific input/candidate hypotheses worth testing;
  20 have the reference available but no reliable selector established; 19
  cannot be resolved from the saved top 10. No fixes or accuracy gain are claimed.
- A post hoc cached-control audit found five top-1 cases with a carbon having
  more than four explicit neighbors and 18 with paired carbon +2/aluminium −2.
  Neither flag appeared in baseline-correct predictions or recorded references.
  Only six flagged rows have the reference within top 10, so their maximum
  oracle headroom is 0.030 percentage points on this prefix.
- Boron cases also expose the evaluator's broad conserved-matrix exception
  bucket. Exact conversion causes were not retained in the original logs.
- No new GPU inference, model fitting, energy computation or untouched-target
  inspection. Further development requires independent train/validation data,
  baseline-correct controls and a frozen held-out comparison.

## Completed label/evaluation audit (2026-10-28)

See [audit report](results/label-audit-20152/REPORT.md),
[case browser](results/label-audit-20152/cases.html), and
[435 unmatched predictions](results/label-audit-20152/unmatched-top1.jsonl).

- Numeric IDs group potentially branching reaction networks, not unique linear
  pathways. Earlier “same pathway” wording should mean “same network ID.”
- Across all 2,834 strict top-1 misses, 2,399 match another recorded target for
  the same normalized input. The earlier 2,381 statistic covered only the
  top-10-hit subset. There are 2,388 exact model-input witness matches overall.
- Strict top-1 accuracy remains 85.94%. Retrospective agreement with any observed
  target for the same input is 97.84%; that diagnostic is not new model accuracy
  or independent confirmation of chemical validity.
- Inputs with one observed target score 97.95% strict top 1; inputs with multiple
  observed targets score 41.46% strict versus 97.46% agreement with any recorded
  target. These cohorts are defined by inspected labels.
- Of 2,399 competing cases, 1,507 share the unique-largest organic fragment
  (proxy available for 2,294). There are 189 unmapped RDKit resonance-related
  comparisons, including three with mapped-H relocation caveats.
- Of 435 unmatched predictions, 343 have the target within top 10 and 92 do not.
  Their chemical validity and reference-label validity remain to be adjudicated.
- This ordered prefix covers only numeric-ID rows; other source tags occur later.
  The full test set is unnecessary for diagnosing the observed ambiguity.
  Broader prespecified sampling is needed before generalizing rates.

No new GPU inference ran. The proposed larger reranker experiment is on hold.
The reserved 512 confirmation cases remain unprocessed; no unprocessed test
target labels were used by the audit.

## Completed failure-learning study (2026-10-28)

See [full report](results/reranking/failure-study/REPORT.md),
[searchable case browser](results/reranking/failure-study/cases.html), and
[all 2,724 isolated cases](results/reranking/failure-study/top1_wrong_top10_correct.jsonl).

- All 20,152 completed rows were audited without dropping unsupported candidates.
- In 2,381 of 2,724 recoverable failures (87.4%), top 1 is another recorded
  target for the same normalized input; 2,370 also share its reaction-network ID.
- 1,814 failures retain the same heavy-atom bonds after removing hydrogen,
  charge, and radical distinctions. These signatures are diagnostic proxies,
  not proven mechanisms or evidence that every alternative is chemically valid.
- Only 343 failures have a top-1 output not observed as another target for that
  input in this prefix. Of these, 207 change heavy connectivity and 70 leave
  the input unchanged; those categories overlap.
- Fixed connected groups split the prefix into 11,933 fitting, 4,277 tuning,
  and 3,942 assessment rows. All remain officially test-source data. Because
  prefix labels informed diagnosis, the assessment is exploratory, not untouched.
- On assessment, frequency scores 3,350/3,942 (84.98%). The simple numerical
  reranker corrects 37 errors and introduces 26 (85.26%, net +11). The
  tuning-selected transformation/ambiguity-aware model corrects 89 and introduces
  85 (85.08%, net +4). Both paired group-bootstrap intervals include zero.
- The primary model corrects 80 but harms 81 on inputs with multiple recorded
  targets. This supports investigating missing branch/context distinctions;
  it does not establish that a stronger reranker will resolve them.
- Frozen model, configurations, split manifest, and all decisions are preserved
  under `results/reranking/failure-study/`. The 512 reserved future test rows
  remain unprocessed and disjoint by input, reaction-network ID, and connected group.

The base checkpoint and inference ranking are unchanged. No physical energy
feature was computed. Full benchmark inference remains paused; the measured
exploratory result does not justify resumption on grounds of reranking benefit.

## Verified saved-rank diagnosis

The raw saved-candidate audit checks every target count against the inference
writer's tally and retains every saved candidate without chemical re-filtering.
Ties use the existing stable order. Results are an ordered prefix, not full-test
or pathway accuracy.

| K | Correct recorded next state within top K | Top 1 wrong but target within top K | Target missing from top K |
|---|---:|---:|---:|
| 1 | 17,318 | 0 | 2,834 |
| 3 | 19,793 | 2,475 | 359 |
| 5 | 20,003 | 2,685 | 149 |
| 10 | 20,042 | 2,724 | 110 |

Of the 2,834 top-1 misses, 86 are ties for the highest sample count. The recorded
target is absent from all saved samples in 105 cases. A mismatch with the recorded
target does not establish that an alternative branch is chemically impossible.
Receipt: `results/reranking/diagnosis-20152/saved-rank-summary.json`.

The original richer extraction stopped at row 19,037 because a saved candidate
SMILES did not roundtrip through RDKit. The new `scripts/diagnose_rank_failures.py`
retains the raw strings and marks that row for baseline fallback. Its complete
`results/reranking/failure-study/records.jsonl` supersedes the truncated old file.
The old file is preserved for provenance and must not be treated as complete.

## Prepared next-step pilot (not yet executed)

Fixed selection in `results/reranking/pilot-v2/selection.json`:

- 512 official training rows in 371 groups, 256 validation rows in 245 groups.
- 512 untouched, unprocessed test rows in 505 groups. All groups connected to
  the diagnostic prefix, validation set, or sampled training pool are excluded.
- Components connect canonical whole-reactant identities or source-local network
  IDs transitively. Sequence indices restart across official splits and are
  therefore namespaced; generic PM/RS/RC/PC labels are ignored as network IDs.
- Selection uses a fixed hash rule before inspecting outcomes. The old prepared
  runner uses K=5; amend and freeze a K=10 protocol to match the latest request
  before running the untouched cohort. All methods must share candidate sets.
- Pairwise logistic compatibility is fitted on training only. Regularization,
  frequency interpolation, switch margin, and optional xTB energy weight are
  selected on validation only. The model is frozen before test inference/scoring.
- Missing physical coverage falls back to frequency. GFN2-xTB is a bounded
  gas-phase, isolated-fragment endpoint-energy diagnostic, not a barrier model.
- Comparison counts corrections, regressions, wrong-to-wrong changes, all-row
  accuracy, top-K oracle coverage, and a paired group-bootstrap interval.
- Prepared resumption rule: validation-selected method only; at least 1 percentage
  point net holdout improvement, positive lower 95% paired group-bootstrap bound,
  at least 30 evaluation groups, and at least 20 discordant outcomes. This is a
  conservative pilot rule, not a formal power calculation or final benchmark claim.

New tools: `scripts/prepare_rerank_pilot.py`, `scripts/pilot_reranker.py`,
`scripts/run_rerank_pilot.py`, and `scripts/summarize_saved_ranks.py`. The xTB adapter
now deduplicates fragments, bounds each calculation, and reports failures without
silently removing evaluation rows. This official-split/physical pilot has not
run; only the separate cached-prefix exploratory study above has been measured.

Do not launch the older runner without adapting its extraction to preserve opaque
candidate strings and checking that its K, frozen protocol, and artifact hashing
match the intended next experiment. The untouched subset remains reserved.

## Preliminary work already performed

These observations concern an ordered 174-row prefix only, not the full test set:

- 149/174 top-1 matches; 174/174 top-3 matches.
- 25 top-3-correct/top-1-wrong rows: 24 strict count deficits and one tie.
- In 24 of those 25 rows, the top-1 candidate is another recorded target for
  the same canonical reactants elsewhere in that prefix.
- 17/25 retain the same heavy-atom graph when protonation/charge is ignored.
- A simple pairwise logistic compatibility baseline fitted on 76 rows from
  15 groups made no changes on a disjoint 55-row, 9-group holdout: both it and
  frequency ranking scored 50/55. This is a small exploratory result.
- Physical scoring did not run: official xTB 6.7.1 downloaded and passed its
  published checksum, but the launcher expected the wrong extracted directory.
  No energy-reranking improvement has been measured. Further pilots are deferred.

Snapshot artifacts are in `results/reranking/pilot-split/`, including
`source-summary.json`, `source-ranking-errors.jsonl`, the fitted baseline, and
`proton-placement-examples.png`. The illustration omits fragments common to the
two candidate systems for readability; scoring must retain every fragment.

## Prepared tools

- `scripts/rerank_analysis.py`: extraction, group-aware exploratory splitting,
  pairwise logistic fitting, and comparison against supplied physical scores.
- `scripts/score_xtb_candidates.py`: bounded GFN2-xTB scoring adapter with
  conformer sampling, charge checks, spin-related abstention, convergence checks,
  and checks for connectivity changes after optimization.
- `tests/test_rerank_analysis.py`: six passing regression checks covering ties,
  canonicalization, alignment failures, absent targets, and group leakage.

The physical adapter sums isolated-fragment electronic energies. This neglects
interfragment interactions, entropy, and activation barriers. Gas-phase scoring
would be an explicit diagnostic assumption, not a validated reaction-condition
model. The learned score is statistical compatibility, not physical energy.

## Scientific interpretation

FlowER ranks predictions by sampling frequency and acknowledges multiple
chemically plausible outcomes under single-recorded-target scoring. Its
published energetic example is a pathway analysis, not evidence that simply
choosing the lowest-energy candidate improves this checkpoint's top-1 accuracy.
See the [author preprint, sections 2.3–2.5](https://arxiv.org/html/2502.12979v1).

Endpoint-energy ranking can penalize necessary uphill intermediates and favor
unchanged reactants. xTB needs geometry, charge, and unpaired-electron inputs;
its total electronic energy is not a reaction barrier or complete Gibbs free
energy. See the [official xTB documentation](https://xtb-docs.readthedocs.io/en/latest/sp.html).

MMFF can assist conformer preparation or strain diagnostics, but its deformation
and nonbonded energy expression should not be interpreted as a general reaction
energy across changing connectivity. See the
[RDKit MMFF implementation paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4116604/).
