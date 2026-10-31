#!/usr/bin/env python3
"""Local, label-bounded Phase 0 audit. No fitting, inference or physical work."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase

from audit_recorded_targets import input_functions

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('results/reranking/full-validation-test20k-20261029')
PREFIX = Path('results/reranking/failure-study/records.jsonl')
INDEX = Path('results/reranking/pilot-v2')
SEED = 'flower-phased-20261031-v1'
HOLDS = ('reranker_training', 'physical_calculations', 'new_inference', 'cloud_rentals', 'reserved_test_outcomes')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class AuditReader:
    """An explicit allowlist; reserved input files containing targets are absent."""
    def __init__(self, paths):
        self.allowed = {Path(p).resolve() for p in paths}
        self.accessed = {}

    def path(self, path):
        path = Path(path).resolve()
        if path not in self.allowed:
            raise PermissionError(f'Not a Phase 0 allowlisted input: {path}')
        return path

    def json(self, path):
        path = self.path(path)
        self.accessed[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def jsonl(self, path):
        path = self.path(path)
        self.accessed[str(path.relative_to(ROOT))] = digest(path)
        with path.open() as stream:
            return [json.loads(line) for line in stream]


def protocol():
    return {
        'version': SEED, 'declared_at': time.time(), 'phase': 0,
        'execution_holds': {key: True for key in HOLDS},
        'minimum_worthwhile_gain_pp': 1., 'primary_k': 10, 'samples': 64,
        'primary_endpoint': 'row-weighted paired strict top-1 accuracy difference',
        'primary_inference': {'alpha': .05, 'sidedness': 'two-sided', 'bootstrap_replicates': 9999,
            'unit': 'connected input/network group', 'interval': 'percentile ratio-of-totals',
            'seed': 20261031, 'every_row_retained': True},
        'interpretation': {'credible_positive': 'paired interval lower endpoint >0',
            'worthwhile_gain_ruled_out': 'paired interval upper endpoint <0.01',
            'inconclusive': 'otherwise; failure to reject zero is not evidence of no useful effect'},
        'power_design': {'gain': .01, 'discordance_rates': [.02, .04, .08],
            'within_group_correlations': [0., .1, .3],
            'n_grid': [1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000],
            'simulations': 500, 'bootstrap_replicates': 499,
            'target_power': [.8, .9], 'seed': 20261031,
            'group_profiles': ['fixed_test_selection', 'validation_sensitivity'],
            'generator': 'Dirichlet-multinomial paired outcomes (+1 rescue, -1 harm, 0 unchanged correctness)',
            'null_check': {'gain': 0., 'n': 20000, 'discordance': .04, 'rho': .3},
            'note': 'Hypothetical effect/discordance/ICC, not a forecast of reranker performance. No reserved labels.'},
        'folds': 5, 'grouping': 'transitive existing input/network components, exact model inputs and normalized inputs',
        'label_catalog': 'validation labels only for primary empirical diagnostics; fitting folds only for lookup',
        'protected_test_manifest': str(EXPERIMENT/'test-rows.json'),
        'not_authorized': list(HOLDS),
    }


def model_input_hash(source, token_function, matrix_function):
    tokens, n = token_function(source); matrix = matrix_function(source)
    if matrix.shape != (n,n) or not np.isfinite(matrix).all():
        raise ValueError('Invalid model input arrays')
    h = hashlib.sha256(b'flower-model-condition-arrays-v1')
    h.update(np.asarray([n],dtype='<i8').tobytes())
    h.update(np.asarray(tokens,dtype='<f8').tobytes())
    h.update(np.asarray(matrix,dtype='<f8').tobytes())
    return h.hexdigest()


def top1(row):
    return row['candidates'][0]['smiles'] if row['candidates'] else None


def label_diagnostics(rows, key):
    catalog = defaultdict(Counter)
    for r in rows:
        catalog[r[key]][r['target']] += 1
    modes = {x: sorted(y for y,n in c.items() if n == max(c.values())) for x,c in catalog.items()}
    counts = Counter(rows=len(rows), inputs=len(catalog))
    counts['best_fixed_choice_matches'] = sum(max(c.values()) for c in catalog.values())
    counts['multi_target_inputs'] = sum(len(c)>1 for c in catalog.values())
    counts['mode_tied_inputs'] = sum(len(m)>1 for m in modes.values())
    for r in rows:
        x, candidate = r[key], top1(r); labels = catalog[x]; mode_count = max(labels.values())
        counts['strict_top1'] += r['target_rank'] == 1
        counts['strict_top10'] += r['target_rank'] is not None and r['target_rank'] <= 10
        counts['any_recorded_top1'] += candidate in labels
        counts['any_recorded_top10'] += any(c['smiles'] in labels for c in r['candidates'])
        counts['multi_target_rows'] += len(labels)>1
        minority = candidate in labels and labels[candidate]<mode_count
        counts['minority_recorded_winner'] += minority
        counts['minority_winner_row_correct'] += minority and candidate == r['target']
        counts['minority_winner_row_wrong'] += minority and candidate != r['target']
        counts['majority_or_comajority_winner'] += candidate in labels and labels[candidate]==mode_count
        counts['unrecorded_winner'] += candidate is not None and candidate not in labels
        counts['empty_pool'] += candidate is None
        counts['a_mode_available_top10'] += any(c['smiles'] in modes[x] for c in r['candidates'])
        counts['target_is_minority_recorded_outcome'] += labels[r['target']]<mode_count
    return dict(counts, best_fixed_choice_accuracy=counts['best_fixed_choice_matches']/len(rows),
                strict_top1_accuracy=counts['strict_top1']/len(rows),
                strict_top10_coverage=counts['strict_top10']/len(rows),
                any_recorded_top1_diagnostic=counts['any_recorded_top1']/len(rows),
                any_recorded_top10_diagnostic=counts['any_recorded_top10']/len(rows),
                fixed_choice_headroom_pp=100*(counts['best_fixed_choice_matches']-counts['strict_top1'])/len(rows),
                catalog='same inspected cohort, including the assessed row; diagnostic only')


def assign_groups(rows, original_metadata):
    parent = list(range(len(rows))); seen = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i,r in enumerate(rows):
        keys = [('base',original_metadata[r['source_row_id']]['split_group']),
                ('exact',r['exact_input']),('normalized',r['normalized_input'])]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i,r in enumerate(rows):
        members[find(i)].append(r)
    for group in members.values():
        identity = hashlib.sha256((SEED+':group:'+str(min(r['source_row_id'] for r in group))).encode()).hexdigest()
        fold = int(identity[:16],16)%5
        for r in group:
            r['phase0_group'] = identity; r['fold'] = fold
    return Counter(len(g) for g in members.values())


def lookup_predictions(fitting, assessed, key):
    catalog = defaultdict(Counter)
    for r in fitting:
        catalog[r[key]][r['target']] += 1
    outcomes = []
    for r in assessed:
        labels = catalog.get(r[key]); choice = top1(r); covered = labels is not None
        used, unavailable = False, False
        if labels:
            m = max(labels.values()); modes = {y for y,n in labels.items() if n==m}
            available = [c['smiles'] for c in r['candidates'] if c['smiles'] in modes]
            if available:
                # Predeclared: frequency winner among co-majorities; no held-out label used.
                choice = available[0]; used = True
            else:
                unavailable = True
        outcomes.append({'source_row_id':r['source_row_id'],'prediction':choice,
            'covered':covered,'used_lookup':used,'mode_unavailable':unavailable,
            'fallback':not used,'correct':choice == r['target'],
            'changed':choice != top1(r),'baseline_correct':r['target_rank']==1})
    return outcomes


def fold_lookup(rows,key):
    out, folds = [], []
    for fold in range(5):
        fitting = [r for r in rows if r['fold']!=fold]
        assessed = [r for r in rows if r['fold']==fold]
        if not fitting or not assessed: raise ValueError('Empty fold')
        for field in ('normalized_input','exact_input','phase0_group'):
            if {r[field] for r in fitting}&{r[field] for r in assessed}:
                raise ValueError('Fold leakage: '+field)
        outcomes = lookup_predictions(fitting,assessed,key); out.extend(outcomes)
        folds.append({'fold':fold,'fitting_rows':len(fitting),'assessed_rows':len(assessed),
                      'fitting_catalog_inputs':len({r[key] for r in fitting}),
                      'lookup_covered':sum(r['covered'] for r in outcomes)})
    return {'rows':len(out),'lookup_covered':sum(r['covered'] for r in out),
        'lookup_used':sum(r['used_lookup'] for r in out),
        'mode_unavailable':sum(r['mode_unavailable'] for r in out),
        'frequency_fallback':sum(r['fallback'] for r in out),'correct':sum(r['correct'] for r in out),
        'changes':sum(r['changed'] for r in out), 'folds':folds,
        'fallback_policy':'original frequency winner, including empty-pool abstention; retain all rows',
        'interpretation':'A leakage-free memorization diagnostic, not an upper bound on learned generalization.'}, out


def meta_summary(rows):
    tags = Counter('numeric_network' if str(r['raw_sequence_id']).isdigit() else r['raw_sequence_id'] for r in rows)
    counts = Counter(r['reactant_key'] for r in rows)
    return {'rows':len(rows),'source_tags':dict(tags),'unique_normalized_inputs':len(counts),
        'rows_with_repeated_input_within_cohort':sum(n for n in counts.values() if n>1),
        'source_row_quantiles':np.quantile([r['source_row_id'] for r in rows],[0,.25,.5,.75,1]).tolist()}


def bias_report(validation, prefix, full_test, selected, val_metadata):
    remainder = [r for r in full_test if r['source_row_id']>=20152]
    cohorts = {'validation':val_metadata,'prefix':full_test[:20152],'remainder':remainder,
               'full_test_input_metadata':full_test,'fixed_test_selection':selected}
    result = {key:meta_summary(value) for key,value in cohorts.items()}
    for name,rows in [('validation',validation),('prefix',prefix)]:
        for metric in ('atoms','heavy_atoms','fragments','abs_charge','charged_atoms','radicals'):
            x = [r['reactant_properties'][metric] for r in rows]
            result[name][metric]={'mean':float(np.mean(x)), 'quantiles':np.quantile(x,[0,.5,.9,.99,1]).tolist()}
    for name,meta in [('fixed_test_selection',selected),('validation',val_metadata)]:
        sizes=np.asarray(list(Counter(r['split_group'] for r in meta).values()))
        result[name]['group_profile']={'groups':len(sizes),'rows':int(sizes.sum()),
            'size_histogram':dict(Counter(map(int,sizes))),
            'size_quantiles':np.quantile(sizes,[0,.5,.9,.99,1]).tolist(),
            'sum_size_squared_over_rows':float(sum(sizes*sizes)/sum(sizes))}
    protected={r['reactant_key'] for r in val_metadata+full_test[:20152]}
    result['selection_guard']={'selected_rows':len(selected),
        'overlapping_normalized_inputs':sum(r['reactant_key'] in protected for r in selected),
        'previously_inspected_selected':sum(r['previously_inspected'] for r in selected),
        'overlapping_groups_selected':sum(r['group_seen_in_validation'] or r['group_touches_inspected_test'] for r in selected),
        'reserved_targets_opened':False}
    assert not any(result['selection_guard'][k] for k in ('overlapping_normalized_inputs','previously_inspected_selected','overlapping_groups_selected'))
    result['limits']=['No reserved target multiplicity, reaction-edit class, stopping labels or correctness inspected.',
        'Full-test chemistry/size comparisons limited by available input-only metadata; raw test file was not opened.',
        'Numeric-network IDs are identifiers, not independent chemical class labels or chronology.',
        'The ordered prefix may reflect serialization, network composition, size and OOM truncation bias.',
        'Fresh group exclusions and random row sampling define the selected estimand; not every full-test row is eligible.']
    return result


def run(out):
    if out.exists(): raise FileExistsError('Preserve earlier artifacts; use a fresh Phase 0 output directory.')
    out.mkdir(parents=True)
    save(out/'protocol.json',protocol())  # Before computing new outcomes.
    paths=[EXPERIMENT/'protocol.json',EXPERIMENT/'validation-rows.json',EXPERIMENT/'test-rows.json',
        EXPERIMENT/'inference/selection.json',EXPERIMENT/'cache/val.jsonl',PREFIX,INDEX/'test-input-index.jsonl',
        Path('results/reranking/cumulative-rarity-20261030/all-audited-rows.jsonl')]
    paths.append(EXPERIMENT/'inference/attempts/val004-0001.json')
    parts=sorted((EXPERIMENT/'inference/records').glob('val[0-9]*.jsonl'))
    paths += parts + [p.with_suffix('.manifest.json') for p in parts]
    reader=AuditReader(paths); original=reader.json(EXPERIMENT/'protocol.json')
    val_meta=reader.json(EXPERIMENT/'validation-rows.json'); test_meta=reader.json(EXPERIMENT/'test-rows.json')
    for name in ('validation-rows.json','test-rows.json'):
        assert reader.accessed[str(EXPERIMENT/name)] == original['manifest_sha256'][name]
    selection=reader.json(EXPERIMENT/'inference/selection.json')
    validation=reader.jsonl(EXPERIMENT/'cache/val.jsonl')
    assert reader.accessed[str(EXPERIMENT/'cache/val.jsonl')]==selection['cached_validation_records_sha256']
    for path in parts:
        receipt=reader.json(path.with_suffix('.manifest.json')); data=reader.jsonl(path)
        assert receipt['status']=='complete' and reader.accessed[str(path)]==receipt['records_sha256']
        validation.extend(data)
    validation.sort(key=lambda r:r['source_row_id']); prefix=reader.jsonl(PREFIX)
    assert [r['source_row_id'] for r in validation]==list(range(21723))
    assert [r['source_row_id'] for r in prefix]==list(range(20152))
    tokenize,matrix=input_functions(ROOT/'utils/data_utils.py'); cache={}
    for split,rows in [('validation',validation),('exploratory_prefix',prefix)]:
        for i,r in enumerate(rows):
            assert r['sample_count']==sum(r['metrics'])==64
            assert (r['target_rank'] is None)==(r['metrics'][0]==0)
            source=r['reaction'].split('>>',1)[0]
            if source not in cache: cache[source]=model_input_hash(source,tokenize,matrix)
            r['exact_input']=cache[source]; r['normalized_input']=r['reactant_key']
            assert hashlib.sha256(r['reactant'].encode()).hexdigest()==r['normalized_input']
            if (i+1)%5000==0: print(split,'input identities',i+1,flush=True)
    groups=assign_groups(validation,{r['source_row_id']:r for r in val_meta})
    summary={'validation':{},'exploratory_prefix':{},'lookup':{},'identity_scope':
        'Exact = ordered token and BE arrays from repository helpers, before stochastic noise; normalized = evaluator no-map/no-stereo explicit-H state.'}
    for split,rows in [('validation',validation),('exploratory_prefix',prefix)]:
        for key in ('exact_input','normalized_input'):
            summary[split][key]=label_diagnostics(rows,key)
        norm_to_exact=defaultdict(set); exact_to_norm=defaultdict(set)
        for r in rows: norm_to_exact[r['normalized_input']].add(r['exact_input']);exact_to_norm[r['exact_input']].add(r['normalized_input'])
        summary[split]['identity_relationship']={'normalized_inputs_with_multiple_exact_arrays':sum(len(x)>1 for x in norm_to_exact.values()),
            'exact_arrays_with_multiple_normalized_states':sum(len(x)>1 for x in exact_to_norm.values())}
    for key in ('exact_input','normalized_input'):
        summary['lookup'][key],outcomes=fold_lookup(validation,key)
        with (out/f'lookup-{key}.jsonl').open('x') as f:
            for row in outcomes: f.write(json.dumps(row)+'\n')
    summary['validation_groups']={'groups':sum(groups.values()),'size_histogram':dict(groups),
                                  'fold_rows':dict(Counter(r['fold'] for r in validation))}
    expanded=reader.jsonl(Path('results/reranking/cumulative-rarity-20261030/all-audited-rows.jsonl'))
    expanded=[r for r in expanded if r['cohort']=='validation']
    assert len(expanded)==21723
    summary['expanded_normalized_catalog_diagnostic']={'rows':len(expanded),
        'any_recorded_top1':sum(r['target_rank']==1 or r['top1_is_recorded_alternative'] for r in expanded),
        'provenance':'previous audited training + full validation + inspected test prefix; not used for fitting/lookup',
        'unmatched_top10_recoverable':sum(r['unmatched_recoverable'] for r in expanded)}
    full_test=reader.jsonl(INDEX/'test-input-index.jsonl')
    assert len(full_test)==218997 and len(test_meta)==20000
    bias=bias_report(validation,prefix,full_test,test_meta,val_meta)
    save(out/'summary.json',summary);save(out/'bias.json',bias)
    attempt=reader.json(EXPERIMENT/'inference/attempts/val004-0001.json')
    elapsed=attempt['finished_at']-attempt['started_at'];generated=attempt['end_rows']-attempt['start_rows']
    entries=[c for r in validation for c in r['candidates']]
    save(out/'feasibility.json',{'validation_rows':len(validation),'saved_top10_candidate_entries':len(entries),
        'entries_with_atom_map_syntax':sum(bool(re.search(r':\d+\]',c['smiles'])) for c in entries),
        'candidate_mapping_status':'No retained generator correspondence in saved candidate strings; mapped edits require a separately verified input/candidate mapping or abstention.',
        'historical_complete_shard':{'part':'val004','rows':generated,'seconds':elapsed,'seconds_per_row':elapsed/generated,
            'projected_20k_gpu_hours_at_this_throughput':elapsed/generated*20000/3600,
            'contingency_2x_hours':elapsed/generated*40000/3600,
            'scope':'Historical A100 80GB shard only; prospective test cost is unmeasured and hardware/chemistry-dependent.'}})
    identities=[{k:r[k] for k in ('source_row_id','exact_input','normalized_input','phase0_group','fold','target_rank')} for r in validation]
    save(out/'validation-identities.json',identities)
    save(out/'provenance.json',{'inputs':reader.accessed,'data_helpers_sha256':digest(ROOT/'utils/data_utils.py'),
        'audit_script_sha256':digest(Path(__file__)),'rdkit':rdBase.rdkitVersion,'numpy':np.__version__,
        'reserved_target_files_opened':[], 'training_started':False,'gpu_used':False,
        'original_protocol_sha256':digest(EXPERIMENT/'protocol.json')})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); run(args.out)
