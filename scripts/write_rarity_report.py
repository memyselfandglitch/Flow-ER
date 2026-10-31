#!/usr/bin/env python3
"""Write the descriptive rarity audit and traceable case illustrations."""
from collections import Counter
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from PIL import Image, ImageDraw, ImageFont

from analyze_candidate_rarity import EXPERIMENT, OLD, read_jsonl, save, sha
from audit_recorded_targets import mapped_change
from rarity_features import signatures, input_fingerprint
from summarize_saved_ranks import target_string

OUT = Path('results/reranking/cumulative-rarity-20261030')
LABELS = ('previous_test', 'validation', 'combined')
NAMES = ('Previous test prefix', 'Full validation', 'Combined observations')
CASES = [
    ('validation', 12988, 'Stopping versus a specific proton transfer',
     'Top 1 leaves methoxide alongside a protonated carboxyl group. The reference transfers that proton to methoxide, giving methanol and the neutral carboxylic acid.',
     'A concrete donor/acceptor and charge-context hypothesis for a selective continuation rule. This is a structural rationale, not experimentally established next-step preference.'),
    ('previous_test', 8045, 'Proton-acceptor choice',
     'Both outcomes deprotonate the same oxygen. Top 1 places the proton on tetrahydrofuran; the reference places it on the tertiary amine in the larger organic fragment.',
     'Test whether candidate-local donor/acceptor context distinguishes these choices. A rarity bonus does not encode this distinction; conditions and competing branches still need review.'),
    ('validation', 8509, 'Protonation site and charge context',
     'Top 1 protonates a carbamate carbonyl oxygen while leaving a tetrazolide anion. The reference protonates that anionic ring nitrogen; both consume one HCl.',
     'A charge- and site-specific hypothesis for review. Do not infer a universal preference for fewer charged atoms.'),
    ('validation', 20059, 'Proton-acceptor choice after elimination',
     'Both outcomes convert a tert-butyl cation to isobutene. Top 1 puts the transferred proton on dioxane; the reference puts it on triethylamine.',
     'Another focused context-selection hypothesis; the shared organic transformation is already predicted. This does not demonstrate a better principal organic product.'),
    ('validation', 8074, 'Unresolved catalyst branch',
     'Top 1 associates Pd with a phosphine. The reference inserts Pd into the substrate C–Cl connection.',
     'This was already an ambiguous branch in the previous pilot review. Low training support does not establish that the reference should outrank the unrecorded association step.'),
    ('previous_test', 643, 'Competing activation and addition steps',
     'Top 1 protonates a carbamate carbonyl oxygen. The reference instead adds hydroxylamine to a ketone, forming a tetrahedral charged adduct.',
     'Both change the supplied mixture differently. The rarity contrast identifies a disagreement but supplies no validated selector for the next step.'),
    ('validation', 12982, 'A rare continuation with only one draw',
     'Top 1 leaves the ester/water mixture unchanged in 63/64 draws. The rank-2 reference is water addition to the ester carbonyl and appears once.',
     'One rare candidate is not evidence that it should be promoted. This case illustrates why a blanket stop penalty or novelty reward needs strong controls.'),
]


def percent(n, d):
    return f'{100*n/d:.2f}%'


def table(headers, body):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                      ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in body])


def draw_case(row, destination):
    # Cancel only identical whole fragments shared by all three states.
    states = []
    ps = Chem.SmilesParserParams(); ps.removeHs = False
    for key in ('reactant', 'top1', 'target'):
        mol = Chem.RemoveHs(Chem.MolFromSmiles(row[key], ps))
        states.append(Counter(Chem.MolToSmiles(f) for f in Chem.GetMolFrags(mol, asMols=True)))
    shared = states[0] & states[1] & states[2]
    mols = []
    for state in states:
        remaining = state - shared
        mol = Chem.MolFromSmiles('.'.join(fragment for fragment, count in remaining.items() for _ in range(count)))
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
    structures = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(600, 430))
    canvas = Image.new('RGB',(1800,510),'white'); canvas.paste(structures,(0,0))
    font_path = Path('/System/Library/Fonts/Helvetica.ttc')
    font = ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    drawing = ImageDraw.Draw(canvas)
    for x,label in zip((300,900,1500),('Input',f'Frequency winner: {row["top1_count"]}/64',
                  f'Recorded target: rank {row["target_rank"]}, {row["target_count"]}/64')):
        drawing.text((x,452),label,fill='black',font=font,anchor='mm')
    drawing.text((900,488),'Unchanged fragments shared by all three states are omitted.',
                 fill='#555555',font=font,anchor='mm')
    canvas.save(str(destination))
    return sum(shared.values())


def additional_summary(rows):
    extra = {}
    for split in LABELS:
        selected = [r for r in rows if split == 'combined' or r['cohort'] == split]
        sh = [r for r in selected if r['rarity_associated_shortlist']]
        robust = [r for r in sh if r['target_support'][1] < 100 and
                  r['top1_support'][1] >= max(100, 10*max(1,r['target_support'][1]))]
        extra[split] = {
            'no_op_reference':sum(r['target_no_op'] for r in selected),
            'rare_recoverable_unmatched':sum(r['unmatched_recoverable'] and not r['target_no_op'] and r['target_support'][0]<100 for r in selected),
            'shortlist_unique_inputs':len({r['input_key'] for r in sh}),
            'shortlist_groups':len({r['analysis_group'] for r in sh}),
            'shortlist_single_observed_target':sum(not r['multiple_recorded_targets'] for r in sh),
            'shortlist_target_rank2':sum(r['target_rank']==2 for r in sh),
            'shortlist_difference_categories':dict(Counter(r['difference_category'] for r in sh)),
            'shortlist_both_radii':len(robust),
            'shortlist_both_radii_non_noop':sum(not r['top1_no_op'] for r in robust),
            'nontermination_with_empty_fingerprint_difference_r1':sum(not r['target_no_op'] and r['target_family'][0]=='no_local_change' for r in selected),
            'nontermination_with_empty_fingerprint_difference_r2':sum(not r['target_no_op'] and r['target_family'][1]=='no_local_change' for r in selected)}
    flagged = [r for r in read_jsonl(OUT/'control-flags.jsonl') if r['flags']]
    extra['distinct_structural_flags'] = {'rows':len(flagged),
        'recoverable':sum(r['target_rank'] is not None and 1<r['target_rank']<=10 for r in flagged),
        'baseline_correct':sum(r['target_rank']==1 for r in flagged)}
    save(OUT/'additional-summary.json',extra)
    return extra


def main():
    s = json.loads((OUT/'summary.json').read_text())
    controls = json.loads((OUT/'controls.json').read_text())['cohorts']
    rows = read_jsonl(OUT/'all-audited-rows.jsonl')
    extra = additional_summary(rows)
    train = json.loads((OUT/'training-index.json').read_text())
    assert sum(train['radius1'].values()) == sum(train['radius2'].values()) == train['training_rows'] == 1933750
    assert not train['parse_failures']
    del train
    lookup = {(r['cohort'], r['source_line']): r for r in rows}
    shortlist = [r for r in rows if r['rarity_associated_shortlist']]
    selected_keys = {(split, line) for split, line, *_ in CASES}
    raw = {}
    sources = [('previous_test', OLD), ('validation', EXPERIMENT/'cache/val.jsonl')]
    sources += [('validation', p) for p in sorted((EXPERIMENT/'inference/records').glob('val[0-9]*.jsonl'))]
    checked = 0
    for split, source in sources:
        for i,line in enumerate(source.open()):
            r = json.loads(line); key = (split, r['source_row_id'] + 1)
            assert r['sample_count'] == sum(r['metrics']) == 64
            assert lookup[key]['target_rank'] == r['target_rank']
            if i % 73 == 0:
                source_state,target_state = r['reaction'].split('>>')
                assert signatures(source_state,target_state) == signatures(r['reactant'],r['target'])
                assert input_fingerprint(source_state) == input_fingerprint(r['reactant'])
                assert target_string(source_state) == r['reactant']
                checked += 1
            if key in selected_keys:
                raw[key] = r
    assert checked == 580
    save(OUT/'normalization-check.json',{'checked_saved_rows':checked,
        'method':'Every 73rd saved row in each source file; raw mapped pair versus evaluator-normalized pair',
        'family_invariance':True,'input_hash_invariance':True,'normalization_exact':True})
    illustrations = []
    dossiers = []
    case_table = []
    for split, line, pattern, observation, interpretation in CASES:
        r = lookup[(split, line)]
        assert r['rarity_associated_shortlist'] and not r['top1_is_recorded_alternative']
        source, target = raw[(split, line)]['reaction'].split('>>')
        dossier = dict(r, pattern=pattern, observation=observation, interpretation=interpretation,
                       reference_edit=mapped_change(source, target),
                       assessment='AI-assisted structural review; not blinded chemist or experimental adjudication')
        if (split, line) in [('validation',12988), ('previous_test',8045)]:
            name = f'case-{split}-{line}.png'
            dossier['shared_fragments_omitted_from_drawing'] = draw_case(r, OUT/name)
            dossier['illustration'] = name
            illustrations.append(f'![{split} line {line}: input, frequency winner and recorded target]({name})')
        dossiers.append(dossier)
        case_table.append([f'{split} line **{line}**', pattern,
            f'{r["top1_count"]}/64 → rank {r["target_rank"]}, {r["target_count"]}/64',
            f'{r["top1_support"][0]:,} → {r["target_support"][0]:,}', observation + ' ' + interpretation])
    save(OUT/'case-review.json', dossiers)
    subset_counts = {}
    for split in LABELS:
        group = [r for r in shortlist if split == 'combined' or r['cohort'] == split]
        single = [r for r in group if not r['multiple_recorded_targets']]
        subset_counts[split] = {'single_observed_target': len(single),
            'single_observed_target_stop_winner': sum(r['top1_no_op'] for r in single),
            'single_observed_target_changed_winner': sum(not r['top1_no_op'] for r in single)}
    save(OUT/'shortlist-single-target-counts.json', subset_counts)

    lines = ['# Candidate coverage and training-pattern rarity audit', '',
        '**Validation generation is complete: 21,723 steps × 64 draws. All 91 remote artifacts were checksum-verified locally, instance 58373066 was cancelled, and the cancellation receipt lists no remaining instances. Reranker training and the new 20,000-step test generation remain held.**', '',
        'The existing pretrained FlowER checkpoint and saved predictions were not changed. This audit uses the full validation candidate set and the already-inspected 20,152-row test prefix. It does not fit a model, change candidate ranking, rent a GPU, or inspect reserved test labels.', '',
        '## Candidate coverage', '',
        table(['Outcome', *NAMES], [[title, *[f'{s["coverage"][split][key]:,}' for split in LABELS]] for title,key in [
            ('Steps','rows'), ('Target at rank 1','top1_correct'), ('Target at ranks 2–10','top10_recoverable'),
            ('Target present below rank 10','target_present_below_top10'), ('Target absent from all 64 draws','target_absent_all64')]]), '',
        'The missing-target rates are **105/20,152 = 0.52%**, **110/21,723 = 0.51%**, and **215/41,875 = 0.51%**. Top-10 misses additionally include the 5 + 4 references below rank 10. Widening K could expose those nine retained references; reordering the existing pool cannot recover the other 215.', '',
        '“Absent from all 64” means the reference was absent from the evaluator’s retained, conserved candidate states after 64 attempts. The preconversion matrices were not preserved here, so this does not distinguish never sampling a reference-like matrix from losing it during conversion. In 213/215 missing-reference rows, some other candidate was retained. Only 12/215 rows had at least 32 draws in the conserved-matrix exception bucket; two lost all 64. That bucket covers multiple conversion/standardization operations and does not identify their exception cause.', '',
        'The unit is a supplied elementary step, not an entire reaction pathway. Combined results are descriptive pooled observations, not a new independent test benchmark. Six normalized inputs and seven input→target transitions occur in both cohorts; uncertainty calculations group related observations. Both inspected cohorts carry numeric-network source tags, so they do not represent every source-tag cohort of the full test set.', '',
        '## Which ranking errors are not already explained by a recorded alternative?', '',
        table(['Subset',*NAMES], [[title,*[f'{s["reranking_opportunity"][split][key]:,}' for split in LABELS]] for title,key in [
            ('Wrong top 1, reference in top 10','recoverable'), ('Top 1 is another recorded target','winner_is_recorded_alternative'),
            ('Top 1 has no recorded target match','winner_not_observed_as_target'), ('Reference is rank 2 in that unmatched subset','unmatched_target_rank2')]]), '',
        '**5,087/5,852 (86.9%)** of the recoverable strict-label mismatches already have a frequency winner recorded as another target for the same normalized input. This reproduces the earlier test-prefix count of 2,381/2,724. The comparison catalog now includes the full training set, full validation, and the inspected test prefix; no unprocessed test labels were searched.', '',
        '**765 rows** remain after excluding those witnessed alternative winners: 420 differ in heavy-atom connectivity, 274 share heavy-atom bonds but differ in hydrogen/charge state, and 71 have other changes with the same heavy connectivity. “No recorded match” does not prove that a candidate is invalid, and an input can still have multiple other recorded targets. Even perfect recovery of these 765 without any new errors would be only a **1.83 percentage-point ceiling** on pooled strict top-1 accuracy, not a predicted gain.', '',
        '## How rarity was measured', '',
        'All **1,933,750 rows of `data/flower_new_dataset/train.txt`** were scanned, with zero parse failures. The earlier 6,875-row sampled training index was not used for support counts. Counts retain repeated training exposures; they are not counts of independent chemistries or measured optimization updates.', '',
        'For each input→candidate transition, we subtract explicit-H sparse Morgan count fingerprints after clearing atom maps. Identical spectator contributions cancel. The primary signature uses radius 1; radius 2 is a sensitivity analysis with more local context. A support count is the number of training rows with the same signed fingerprint difference. These are hashed structural-change proxies, **not exact reaction templates, named mechanism frequencies, or proof of chemical validity**. In 17 nontermination observations at radius 1 (nine at radius 2), a changed reference has an empty fingerprint difference, illustrating the proxy’s limited resolution.', '',
        'The exploratory low-support definition is **fewer than 100 training rows**; the full support bands are also reported. These counts are distinct from the candidate’s frequency among 64 Monte Carlo draws. The radius-1 unchanged-state signature occurs in 348,742 training rows, so stopping is treated separately and no-change references are excluded from rarity-rate comparisons.', '',
        '## Association between rarity and errors', '',
        'First restrict to nontermination steps whose normalized input has **only one observed target** in the searched label catalog. This reduces the known competing-label problem; it cannot rule out unrecorded alternatives.', '',
        table(['Cohort','Rare: errors / rows','Common: errors / rows','Difference, percentage points (grouped 95% interval)'], [
            [name, f'{round(s["associations"][split]["radius1"]["single_observed_target"]["rare_error_rate"]*s["associations"][split]["radius1"]["single_observed_target"]["rare_rows"]):,} / {s["associations"][split]["radius1"]["single_observed_target"]["rare_rows"]:,} ({100*s["associations"][split]["radius1"]["single_observed_target"]["rare_error_rate"]:.2f}%)',
             f'{round(s["associations"][split]["radius1"]["single_observed_target"]["common_error_rate"]*s["associations"][split]["radius1"]["single_observed_target"]["common_rows"]):,} / {s["associations"][split]["radius1"]["single_observed_target"]["common_rows"]:,} ({100*s["associations"][split]["radius1"]["single_observed_target"]["common_error_rate"]:.2f}%)',
             f'{s["associations"][split]["radius1"]["single_observed_target"]["difference_pp"]:.2f} ({s["associations"][split]["radius1"]["single_observed_target"]["group_bootstrap_95ci_pp"][0]:.2f} to {s["associations"][split]["radius1"]["single_observed_target"]["group_bootstrap_95ci_pp"][1]:.2f})'] for split,name in zip(LABELS,NAMES)]), '',
        'The combined radius-1 association is **4.73% versus 1.23%** (about 3.85×), with a 3.50-point difference. Atom-count-band standardization gives 3.53 points. Radius 2 retains the same direction in this single-observed-target subset: **3.04% versus 1.50%**, difference 1.53 points (grouped interval 1.05–2.04).', '',
        '**The unrestricted association is definition-sensitive.** Across all nontermination inputs, radius 1 gives 21.09% versus 14.88%, but radius 2 reverses to 17.34% versus 19.14%. After standardizing for observed target multiplicity and atom-count bands, the radius-2 difference changes to +1.51 points. This is a confounding/aggregation warning: an unqualified claim that “rarity causes failures” is not supported.', '',
        table(['Radius-1 training support','Single-target nontermination rows','Top-1 errors','Missing target','Recoverable top-10 errors'], [
            [r['band'],f'{r["rows"]:,}',f'{r["top1_wrong"]:,} ({100*r["top1_error_rate"]:.2f}%)',r['target_absent_all64'],r['top10_recoverable']]
            for r in s['rarity']['combined']['radius1']['single_observed_target']]), '',
        'Rarity also tracks missing candidates: **187/215** all-pool misses have a rare, nontermination reference by the radius-1 proxy (99 previous test + 88 validation). Of the 206 nontermination misses, 187 are rare; there are also nine missed no-change references. The rare/nonrare pool-miss rates among all nontermination steps are **187/15,467 = 1.21%** versus **19/18,719 = 0.10%**. Those are generator/conversion coverage problems, not current-pool reranking opportunities.', '',
        '## Isolated rarity-associated ranking cases', '',
        'The stricter review shortlist requires: reference at ranks 2–10; top 1 not another recorded target; no reference no-op or tie for first; reference support <100; and winner support ≥100 and ≥10×max(1, reference support). The thresholds are exploratory diagnostic definitions, not fitted selectors.', '',
        table(['Subset',*NAMES],[
            ['Unmatched recoverable errors with rare nontermination reference',*[extra[x]['rare_recoverable_unmatched'] for x in LABELS]],
            ['Stricter rare-target/common-winner shortlist',*[s['reranking_opportunity'][x]['rarity_associated_shortlist'] for x in LABELS]],
            ['Winner stops; reference continues',*[s['reranking_opportunity'][x]['rarity_associated_shortlist']-s['reranking_opportunity'][x]['rarity_shortlist_excluding_baseline_no_op'] for x in LABELS]],
            ['Winner and reference both change the input',*[s['reranking_opportunity'][x]['rarity_shortlist_excluding_baseline_no_op'] for x in LABELS]],
            ['Shortlist with only one observed target',*[subset_counts[x]['single_observed_target'] for x in LABELS]],
            ['Shortlist passing both radius definitions',*[extra[x]['shortlist_both_radii'] for x in LABELS]]]), '',
        'The **156** shortlisted observations cover **144 inputs and 115 analysis groups**, not 156 independent mechanisms. The reference is rank 2 in 104 cases. **122** have only one observed target; the other 34 have multiple recorded targets, although their particular top-1 winner was not found among them. Requiring the rarity contrast at both radii leaves **130** cases, including **17** where both candidates change the input. An oracle fixing all 156 with no harms would add only 0.37 percentage points overall.', '',
        '### Specific cases and what a selector would need to learn', '',
        'These seven cases were chosen to illustrate different interpretations, not sampled to estimate their prevalence. Full inputs, candidate strings, training support at both radii, mapped reference edits, and review qualifications are preserved in [case-review.json](case-review.json). None is a chemist-adjudicated correction.', '',
        table(['Source (one-based line)','Pattern','Sample counts','Training support: winner → reference (radius 1)','Observed difference and interpretation'],case_table), '',
        *illustrations, '',
        '## Baseline-correct controls prevent a rarity-only rule', '',
        'Apply a label-blind flag: a common winner (support ≥100) has at least one lower-ranked top-10 candidate with <100 support and ≥10-fold less support. This only marks cases; no model was fitted and no outputs were reordered.', '',
        table(['Flagged population',*NAMES],[[title,*[controls[x]['rare_alternative_gate'][key] for x in LABELS]] for title,key in [
            ('Flagged steps','flagged_rows'),('Baseline already correct','baseline_correct'),('Baseline wrong','baseline_wrong'),
            ('Recorded target among flagged alternatives','target_among_flagged_alternatives'),
            ('Same, excluding witnessed alternative winner','unmatched_target_among_flagged_alternatives')]]), '',
        '**6,836/8,786 (77.8%) flagged predictions are already correct.** Always replacing them with a rarer alternative would introduce 6,836 strict-label errors, while even an oracle could rescue at most 538 strict labels within that flag set; only 156 of those rescues avoid a witnessed alternative winner. This is an algebraic bound for that hypothetical blanket policy, not measured reranker performance. It directly argues against rewarding rarity by itself.', '',
        'As a separate, specific validity check, carbon with >4 explicit neighbours appears in **seven** wrong top-1 candidates (five test, two validation), zero correct winners, and zero references; all seven references are in top 10. The combined carbon-positive/aluminium-negative flag appears in another **27** wrong winners, but only one reference is in top 10. The union is **34 flagged wrong rows, eight recoverable**, with no flagged correct winners in these inspected data. These narrow post hoc flags do not establish general chemical validity, future zero false-positive rates, or identify the correct replacement by themselves.', '',
        '## Practical interpretation', '',
        'The data support using training-pattern rarity to **stratify diagnosis and select examples for review**. They do not establish rarity as the cause of any individual failure or show that a rarity-weighted reranker improves accuracy. The most actionable hypotheses are specific donor/acceptor or site choices, identifiable local valence defects, and justified continuation versus stopping. Plausible unrecorded catalyst branches and uncertain references require adjudication rather than automatic promotion.', '',
        'If training is later authorized, begin with the validation-only shortlist (89 cases, 73 with one observed target), review which references are chemically defensible, and learn the contextual distinction alongside matched negatives and baseline-correct controls. Include rare correct predictions and valid terminal states. Treat missing targets as a separate generation/conversion task. Freeze any learned method before the already-selected untouched 20,000-step test comparison, and report both rescues and new errors. **Training remains held.**', '',
        '## Artifacts and verification', '',
        '- [All 215 missing-target cases](missing-from-all-samples.jsonl).',
        '- [All 5,852 recoverable strict-label errors](recoverable-cases.jsonl).',
        '- [The 765 without a recorded alternative winner](unmatched-recoverable-cases.jsonl).',
        '- [The 156 rarity-associated cases](rarity-associated-shortlist.jsonl).',
        '- [Coverage, support bands, subgroup associations and intervals](summary.json), [controls](controls.json), [additional counts](additional-summary.json).',
        '- [Source hashes](sources.json), [full training-support index](training-index.json), [580-row normalization check](normalization-check.json).', '',
        'Verification: all 41,875 rows have unique source-row identities within cohort and exactly 64 draws; global target ranks agree with recorded correct-sample counts. Independent coverage and shortlist membership checks agree. Both training-index count sums equal 1,933,750 with zero failures. Four feature-invariance tests passed, 580 saved pairs preserved their signatures under evaluator normalization, and the two earlier structural-control counts were independently reproduced. The checkpoint, source predictions, and frozen experiment protocol remain unchanged.', '',
        'Reproduce from the repository root:', '', '```bash',
        '.venv-analysis/bin/python scripts/analyze_candidate_rarity.py --out results/reranking/cumulative-rarity-20261030 --workers 6',
        '.venv-analysis/bin/python scripts/audit_rarity_controls.py --out results/reranking/cumulative-rarity-20261030',
        '.venv-analysis/bin/python scripts/write_rarity_report.py', '```', '',
        '## Material passport and statistical scope', '',
        'academic-research-suite / experiment-agent / validate; 2026-10-30; verification status **ANALYZED**. Local saved-data audit with source/invariant checks, not an independent benchmark reproduction or a trained reranker experiment. Case review is AI-assisted and not independent human chemical adjudication.', '',
        'Uncertainty uses 2,000 seeded bootstrap resamples of connected groups sharing normalized inputs or source-specific numeric network IDs. These are descriptive 95% intervals, not simultaneous multiple-comparison-adjusted intervals. Thresholds and feature definitions are exploratory. Size/multiplicity standardization leaves other confounders; finite sample frequency, training diversity, label provenance, reaction context, conversion behavior and missing conditions may all matter.', '',
        'All 11 statistical-fallacy categories were checked: (1) aggregation/Simpson reversal is explicitly reported for radius 2; (2) no group-level association is treated as an individual cause; (3) ordered-prefix and diagnostic-subset selection limit generalization; (4) target-multiplicity conditioning is descriptive, not a causal adjustment; (5) correct-baseline prevalence is included; (6) there is no before/after improvement claim; (7) all rows, misses and empty pools are retained; (8) all radii/cohort/subgroup comparisons are preserved without selective significance claims; (9) exploratory thresholds and both context radii are disclosed; (10) association is not causation; (11) training support precedes generation, but this does not identify a causal direction or intervention effect.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    save(OUT/'report-provenance.json', {'analysis_script_sha256':sha(Path('scripts/analyze_candidate_rarity.py')),
        'controls_script_sha256':sha(Path('scripts/audit_rarity_controls.py')),
        'report_script_sha256':sha(Path(__file__)), 'case_count':len(dossiers),
        'scope':'No candidate generation or model fitting; no reserved test-target access.'})
    print('Wrote report and',len(dossiers),'qualified case reviews')


if __name__ == '__main__':
    main()
