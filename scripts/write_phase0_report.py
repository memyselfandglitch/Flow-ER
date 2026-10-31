#!/usr/bin/env python3
"""Render the completed Phase 0 statistics without accessing test targets."""
import argparse
import json
import math
from pathlib import Path
import shutil


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def pct(n,d):return f'{100*n/d:.3f}%'


def main(root):
    if not (root/'IMPLEMENTATION_PLAN.md').exists():
        shutil.copyfile(Path(__file__).resolve().parents[1]/'PHASED_RERANKING_PLAN.md',root/'IMPLEMENTATION_PLAN.md')
    s=json.loads((root/'summary.json').read_text());b=json.loads((root/'bias.json').read_text())
    p=json.loads((root/'power.json').read_text());f=json.loads((root/'feasibility.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    v=s['validation'];x=v['exact_input'];z=v['normalized_input'];n=z['rows']
    metrics=[('Strict top 1','strict_top1'),('Strict top-10 coverage','strict_top10'),
             ('Any recorded target: top 1 (diagnostic)','any_recorded_top1'),
             ('Any recorded target: top 10 (diagnostic)','any_recorded_top10'),
             ('Best fixed choice per input (in-sample)','best_fixed_choice_matches')]
    metric_rows=[[name,f'{x[key]:,}/{n:,} ({pct(x[key],n)})',f'{z[key]:,}/{n:,} ({pct(z[key],n)})'] for name,key in metrics]
    requirements=[]
    for q in (.02,.04,.08):
        rr={r['target_power']:r for r in p['required_sample_grid'] if r['discordance']==q and r['rho']==.3}
        at20=next(r for r in p['scenarios'] if r['profile']=='fixed_test_selection' and r['nominal_rows']==20000 and r['delta']==.01 and r['discordance']==q and r['within_group_rho']==.3)
        requirements.append([f'{100*q:.0f}%',f'{100*(q+.01)/2:.1f}% / {100*(q-.01)/2:.1f}%',
            f'{rr[.8]["first_grid_n_mc_lower_bound"]:,}',f'{rr[.9]["first_grid_n_mc_lower_bound"]:,}',
            f'{100*at20["positive_detection_power"]:.1f}% ({100*at20["power_mc_95ci"][0]:.2f}–{100*at20["power_mc_95ci"][1]:.2f}%)'])
    minimum=math.ceil(n*.01)
    text=['# Phase 0 — headroom and feasibility audit','',
        '**Recommendation: GO for an explicitly authorized, bounded CPU Phase 1 linear pilot; NO-GO for physical spending, new inference or cloud rental at this gate.** A one-percentage-point strict gain is statistically testable and fits inside observed input-only headroom, but the evidence does not show that it is likely. No model was trained and no execution hold was lifted.', '',
        'This audit uses the completed **21,723 official-validation candidate sets**. The inspected **20,152-row test prefix** is exploratory development evidence. The separate fixed **20,000-row test selection remains unchanged and its targets/outcomes were not opened**. Only its existing input/network metadata was read for bias and power planning.', '',
        '## Accuracy, label agreement and empirical headroom', '',
        table(['Validation metric','Exact model-condition input','Normalized input'],metric_rows), '',
        f'Unique inputs: **{x["inputs"]:,} exact array identities** versus **{z["inputs"]:,} normalized states**. Exact identity hashes the unpadded element-token array and bond/electron matrix computed by the actual repository helpers, before random generation noise. Normalized identity uses the evaluator’s explicit-H, no-map/no-stereo full-mixture state. Thirty-six normalized states have multiple exact array representations; no exact identity combines different normalized states in these validation data.', '',
        'The requested empirical best fixed choice is computed directly as `sum_x max_y count(x,y) / total_rows`, separately under both definitions. It leaves **502 rows / 2.311 points** of headroom for exact inputs and **492 rows / 2.265 points** for normalized inputs above frequency ranking. This describes the observed labels and assumes access to their full in-sample counts. It is not an achievable prediction guarantee, a chemical-validity bound, an estimate of population outcome probabilities, or a generalization upper bound. It also differs from the target-aware per-row top-10 oracle: candidate pools may differ between repeated inputs, and a candidate-aware rule can behave differently on those pools.', '',
        'The primary any-recorded-target catalog consists only of labels within the same full validation cohort, including the assessed row. It is explicitly an in-sample label-set diagnostic. The pre-existing expanded normalized catalog (official training + validation + inspected prefix) gives the same top-1 count, 21,197; it is not passed to fitting folds. Agreement with a recorded label does not establish experimental chemical validity.', '',
        '## Minority outcomes and ties', '',
        table(['Validation diagnostic','Exact inputs','Normalized inputs'],[
            ['Inputs with multiple recorded targets',x['multi_target_inputs'],z['multi_target_inputs']],
            ['Inputs tied for largest label count',x['mode_tied_inputs'],z['mode_tied_inputs']],
            ['Rows on multi-target inputs',x['multi_target_rows'],z['multi_target_rows']],
            ['FlowER winner has strictly lower recorded count than the mode',x['minority_recorded_winner'],z['minority_recorded_winner']],
            ['Those minority winners are correct for their own row',x['minority_winner_row_correct'],z['minority_winner_row_correct']],
            ['Those minority winners are wrong for their own row',x['minority_winner_row_wrong'],z['minority_winner_row_wrong']],
            ['Winner is not any recorded validation target',x['unrecorded_winner'],z['unrecorded_winner']],
            ['Empty candidate pool',x['empty_pool'],z['empty_pool']]]), '',
        '“Majority” here means highest observed count (mode/plurality), not necessarily more than half of labels. Tied modes are not minority outcomes. There are **zero exact-input minority selections** and only **nine normalized-input selections (0.041% of all validation rows; 0.191% of multi-target rows)**. Two of those nine are already correct for their row. With 1,939 of 1,941 ambiguous exact-input groups tied, branch ambiguity largely reflects equally recorded alternatives. Promoting a supposedly more common recorded branch is therefore not a general solution.', '',
        '## Fold-local majority lookup', '',
        'The lookup baseline builds label counts from each fitting partition only, selects an available modal outcome using original frequency order for co-mode ties, and falls back to the original frequency winner when the input is unseen or no fitting mode is in the candidate pool. It cannot insert a candidate. Every assessment row remains in the denominator.', '',
        table(['Five-fold validation lookup','Exact inputs','Normalized inputs'],[
            ['Lookup coverage',f'{s["lookup"]["exact_input"]["lookup_covered"]}/{n}',f'{s["lookup"]["normalized_input"]["lookup_covered"]}/{n}'],
            ['Lookup used',0,0],['Frequency fallback',n,n],['Changed predictions',0,0],
            ['Correct / all rows','18,481 / 21,723','18,481 / 21,723'],['Accuracy','85.076%','85.076%']]), '',
        'This zero coverage is the expected consequence of grouping identical inputs into the same fold. The 2,873 connected groups keep normalized states, exact arrays and source-related network components together. Outer fold sizes are 4,487, 4,462, 4,264, 3,960 and 4,550. Assertions reject cross-fold sharing. **A memorization lookup with no held-out keys is not an upper bound on a learned model’s ability to generalize chemical distinctions.** It is not the reason for the cost gate or the conditional recommendation.', '',
        '## Prefix bias and representativeness', '',
        table(['Available metadata','Inspected prefix','Full validation','Fixed test selection'],[
            ['Rows','20,152','21,723','20,000'],['Distinct normalized inputs','17,628','18,922','19,654'],
            ['Rows sharing an input with another row in that cohort','4,380','4,773','671'],
            ['Numeric-network source rows','20,152','21,723','19,773'],
            ['Other source-tag rows','0','0','227'],
            ['Mean atoms including explicit H',f'{b["prefix"]["atoms"]["mean"]:.2f}',f'{b["validation"]["atoms"]["mean"]:.2f}','Not available in inspected input index'],
            ['Mean mixture fragments',f'{b["prefix"]["fragments"]["mean"]:.2f}',f'{b["validation"]["fragments"]["mean"]:.2f}','Not available in inspected input index'],
            ['Maximum atoms / fragments','398 / 19','418 / 26','Not inspected']]), '',
        'The prefix is the first 20,152 rows, not a random reaction sample. It and validation contain only numeric-network source tags. The full test input index has 2,429 other-tag rows out of 218,997 (1.109%): RC 136, PC 1,653, PM 294 and RS 346. The fixed selection includes RC 13, PC 150, PM 31 and RS 33 (227/20,000 = 1.135%). Generic tags do not identify distinct related reaction networks; dependence beyond recorded input/network links remains uncertain.', '',
        'Prefix and validation have similar central input sizes, but validation has a heavier upper tail (99th-percentile atoms 290 versus 266, and maximum 26 versus 19 fragments). Similar input summaries do not establish similar mechanism classes, label quality or error rates. Prefix strict accuracy is 85.937%, versus 85.076% for validation; its normalized fixed-choice statistic is 87.852%, versus 87.341%. These are descriptive cohort differences.', '',
        'The reserved selection was already frozen as a random row sample from 198,712 eligible rows after excluding groups touching validation or the inspected prefix. It contains no normalized-input/group overlap with those development cohorts. Its repeated-input count is **not label multiplicity**. We did not open its targets to learn whether repeated inputs have one or several outcomes. We also did not open the raw test file to derive structural features, reaction edits or stopping labels. Sparse row sampling changes the observed group-size/repetition profile, and exclusions alter the estimand. Neither this prefix nor validation demonstrates representativeness across chemical mechanisms, rare classes, conditions or dataset provenance.', '',
        '## Minimum worthwhile gain and paired power', '',
        f'The provisional worthwhile gain was fixed at **+1.00 absolute percentage point**, before new audit calculations, in [protocol.json](protocol.json). This means at least **{minimum} net validation corrections**, or **200 net corrections on the fixed 20,000 test rows**. The primary test uses a two-sided 95% paired group-bootstrap interval; detecting a true one-point gain against zero is different from proving that the gain exceeds one point.', '',
        'Power simulations use paired outcomes `+1=correction`, `−1=new error`, `0=unchanged correctness`. A Dirichlet-multinomial group generator sets the expected net gain to 0.01, discordance (corrections + new errors) to 2%, 4% or 8%, and within-group correlation to 0, 0.1 or 0.3. Wrong-to-wrong switches have zero paired difference, so intervention rate can exceed discordance. The prior 1,024-row pilot had 29 discordant rows (2.83%); this motivates a range, not a forecast of the new model.', '',
        'The 20,000-row simulations use the **actual 12,931 group sizes from input metadata**, with median one, maximum 11, and `sum(n_g²)/sum(n_g)=2.123`. Other sample sizes draw complete groups from that profile and record the actual denominator. Each scenario has **500 simulated experiments**, each assessed by **499** pairs group-bootstrap replicates; identical `(group size, paired total)` groups are compressed exactly for computation. Bootstrap estimates always use total paired changes divided by total resampled rows. Frozen evaluation will use **9,999** replicates. No test labels or hypothetical learned scores were used.', '',
        table(['Discordance','Expected corrections / new errors','Rows for 80% power','Rows for 90% power','Estimated power at 20k (MC 95% interval)'],requirements), '',
        'This conservative table uses the largest simulated within-group correlation (0.3) and the first tested sample size whose **Monte Carlo lower confidence bound** reaches the target power. These are grid estimates, not exact minimum sample sizes. Across all nine fixed-selection scenarios, 20,000-row point estimates range from **99.6% to 100%**; observing 100% in 500 simulations does not establish certainty.', '',
        'Grouping matters: resampling validation-like groups instead (mean size 7.56, `sum(n_g²)/sum(n_g)=12.506`) gives only **65.6% power** near 20,000 rows for 8% discordance and correlation 0.3, versus 99.6% with the fixed test selection’s profile. With 4% discordance the validation-profile sensitivity gives 94.2%. This does not change the actual reserved selection, but exposes the dependence assumption. The two zero-effect calibration scenarios have two-sided rejection rates 5.0% and 5.6%. If outer-development discordance/dependence or new input-only group links fall outside the assumed range, update power before test unsealing; do not infer power from the row count alone.', '',
        'A nonsignificant estimate is inconclusive when its interval still includes a one-point benefit. An upper interval endpoint below +1.00 point rules out that worthwhile gain at the stated interval level, even if a smaller effect is positive. A point estimate above one point with a lower endpoint above zero supports a positive result whose estimate meets the target, but does not prove that the true gain exceeds one point.', '',
        '## Chemical feasibility and cost gate', '',
        'Of 3,128 validation top-10-recoverable strict errors, **2,706 frequency winners are already other recorded targets**. The remaining **422** are not matched to another target in the searched catalog: 213 differ in heavy connectivity, 162 share heavy bonds with H/charge differences, and 47 have other changes on the same heavy connectivity. These are structural comparisons, not confirmed mechanisms or chemically proven failures. Their row-wise oracle headroom is **422/21,723 = 1.943 points**.', '',
        f'To gain one point without relying on switching between recorded alternatives requires **{minimum}/422 = {100*minimum/422:.1f}%** of this pool with zero new errors. With 50 new errors it needs 268 corrections (63.5%); with 100 it needs 318 (75.4%). Those are requirements, not forecasts. The existing review identifies specific proton-acceptor/site choices and local validity issues worth testing, but not hundreds of adjudicated corrections.', '',
        'The validation rarity shortlist has 89 cases (73 single-observed-target inputs); even perfect recovery of all 89 without harms adds only 0.410 points. The two validation carbon-overcoordination cases are still smaller. Rarity itself must not receive a reward, and same-heavy-bond differences cannot be called verified proton transfers. The earlier small pilot’s +0.879-point result had a −0.100 to +1.898-point interval and many switches between recorded alternatives; it does not establish a new chemically grounded one-point gain.', '',
        f'There are **{f["saved_top10_candidate_entries"]:,} saved top-10 candidate entries and {f["entries_with_atom_map_syntax"]} with atom-map syntax**. Candidate generation correspondence was not retained in these strings. A first linear model can operate on full-mixture/map-free descriptors; mapped edits and verified-transfer strata need a separately validated correspondence route or an explicit unavailable category. Reference atom maps cannot supply that route.', '',
        'Phase 1 can use already-cached validation data on a local CPU; no GPU is needed for the proposed linear fitting. A proposed initial cap is two CPU-hours after explicit authorization. Phase 0 used local CPU analysis/simulation only and made no cloud calls or physical calculations.', '',
        f'Test candidate generation would be the expensive step: the last completed A100 shard generated 3,803 rows in {f["historical_complete_shard"]["seconds"]/3600:.2f} hours, or {f["historical_complete_shard"]["seconds_per_row"]:.2f} seconds/row. A simple extrapolation is **{f["historical_complete_shard"]["projected_20k_gpu_hours_at_this_throughput"]:.1f} GPU-hours** for 20,000 rows, with a twofold contingency of **{f["historical_complete_shard"]["contingency_2x_hours"]:.1f} hours**. This is a historical throughput illustration, not a measured test runtime or current rental quote. Chemical complexity, hardware, batching and failures may change it. Obtain a fresh price/budget only if rental is later authorized.', '',
        '**Go/no-go:** observed headroom and identifiable distinctions justify a bounded, explicitly authorized linear development experiment. Majority-lookup noncoverage is not a reason to stop. However, the attainable chemical benefit remains uncertain and the one-point target is ambitious. Do not incur new test-generation or physical-computation cost now. Require the development evidence and a separately approved budget before those steps. Physical features get their own coherent-distinction/opportunity/cost gate; a nonsignificant linear result alone neither approves nor rejects them. Zero intervention remains selectable.', '',
        '## Deliverables, commands and checks', '',
        'Next: [the phased implementation plan](IMPLEMENTATION_PLAN.md). It specifies the additive score, nested fold-local fitting, positive-label treatment, bounded weighting, correspondence rules, physical ablations/fallback, frozen assessment and working commands. Later trainers/runners are specified but are not represented as implemented or authorized.', '',
        '- [Metric and lookup audit](summary.json), [input/network bias audit](bias.json), [power results and sample-size grid](power.json), [power plan](power-plan.json).',
        '- [Protected-read provenance](provenance.json), [validation identities and grouped folds](validation-identities.json), [mapping and historical runtime feasibility](feasibility.json).',
        '- [Exact-input lookup outcomes](lookup-exact_input.jsonl) and [normalized-input lookup outcomes](lookup-normalized_input.jsonl).',
        '- Code: `scripts/phase0_reranking_audit.py`, `scripts/phase0_group_power.py`, `scripts/phased_reranking_contracts.py`, `scripts/phased_reranking_gate.py`; tests: `tests/test_phased_reranking.py`.', '',
        'The 22 focused tests check identity distinctions, grouped leakage barriers, fixed-choice/tie arithmetic, fitting-label-only lookup, candidate-availability fallback, reserved-target read rejection, held execution without writes, row-weighted group-bootstrap math, compressed resampling equivalence, paired simulation conservation, additive residual score/margin/zero policy, soft positives, bounded weights and correspondence/provenance checks. No test fits a reranker.', '',
        '## Statistical scope and material passport', '',
        'academic-research-suite / experiment-agent / validate; 2026-10-31. This is a local Phase 0 audit and planning simulation, not a trained-model result. Group-power figures are conditional on hypothetical effect/discordance/correlation and known group sizes; they are not predicted effectiveness. Existing records and frozen selection manifests are preserved. Read receipts enumerate every audited data source; no reserved target file is included.', '',
        'All 11 fallacy categories were reviewed: aggregation effects are addressed by exact/normalized and cohort reporting; no group association proves individual chemistry; prefix/exclusion selection limits generalization; diagnostic ambiguity restrictions are not causal adjustments; correct-baseline prevalence and new-error rates are retained; no before/after training improvement is claimed; missing pools/fallback rows remain; all 83 power scenarios are preserved; grids and thresholds were declared before simulation; rarity is not treated as causal; historical training support cannot establish the causal source of an error. Multiple power scenarios are sensitivity analyses, not searches for a significant model result. Sparse chemical subgroups need their group counts and uncertainty disclosed.', '']
    if (root/'reproduction-verification.json').exists():
        text.extend(['Verification status: **VERIFIED for the deterministic audit**. A fresh-directory rerun reproduced the metric summary, bias summary, identity/fold manifest, both lookup outcome files and feasibility statistics exactly. Both runs’ provenance is preserved in [the reproduction receipt](reproduction-verification.json). Power is a conditional simulation with Monte Carlo uncertainty, not verified model performance.',''])
    (root/'PHASE0_AUDIT.md').write_text('\n'.join(text))
    print('Wrote',root/'PHASE0_AUDIT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args();main(a.audit)
