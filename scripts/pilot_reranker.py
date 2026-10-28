#!/usr/bin/env python3
"""Train/validate a reranker, then compare once on a frozen test cohort.

Records use rerank_analysis.analyze_row's schema plus source_split and split_group.
Real sequence IDs must be namespaced by their original dataset before grouping.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rerank_analysis import FEATURES, canonical, components

VERSION = 'flower-reranking-pilot-v1'
GENERIC_SEQUENCE_IDS = {'', 'PM', 'RS', 'RC', 'PC'}
REGULARIZATION = (0.01, 0.1, 1.0)
ALPHAS = (0.1, 0.25, 0.5, 0.75, 1.0)
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0)
ENERGY_WEIGHTS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
HARTREE_TO_KCAL_MOL = 627.509474
BOOTSTRAP_SEED = 20261028
BOOTSTRAP_RESAMPLES = 10000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_signature(rows):
    return {'reactants': sorted({canonical(row['reactant']) for row in rows}),
            'sequences': sorted({str(row['sequence_id']) for row in rows} - GENERIC_SEQUENCE_IDS),
            'groups': sorted({row['split_group'] for row in rows})}


def assert_disjoint(left, right, context):
    for field in ('reactants', 'sequences', 'groups'):
        if set(left[field]) & set(right[field]):
            raise ValueError(f'{context}: leakage through shared {field}')


def validate_records(rows, expected_split):
    if not rows:
        raise ValueError(f'Empty {expected_split} dataset')
    seen, canonical_rows = set(), []
    for row in rows:
        if row.get('source_split') != expected_split:
            raise ValueError(f'Expected source_split={expected_split}; never fit/tune on test rows')
        key = row.get('row_id')
        if isinstance(key, bool) or not isinstance(key, (str, int)) or str(key) in seen:
            raise ValueError('row_id must be a unique integer or string within each dataset')
        seen.add(str(key))
        if not isinstance(row.get('split_group'), str) or not row['split_group']:
            raise ValueError('Every row requires a frozen split_group')
        if not isinstance(row.get('sequence_id'), str):
            raise ValueError('Every row requires a string sequence_id')
        samples = row.get('sample_count')
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError('sample_count must be a positive integer')
        reactant, target = canonical(row['reactant']), canonical(row['target'])
        canonical_rows.append(dict(row, reactant=reactant))
        previous, candidates_seen, total = math.inf, set(), 0
        for rank, candidate in enumerate(row['candidates'], 1):
            count = candidate['count']
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count > previous:
                raise ValueError('Candidates must retain descending positive integer frequency order')
            previous = count
            total += count
            if candidate['rank'] != rank:
                raise ValueError('Candidate ranks must preserve original stable frequency ordering')
            smi = canonical(candidate['smiles'])
            if smi in candidates_seen:
                raise ValueError('Equivalent candidates must be merged before fitting or evaluating')
            candidates_seen.add(smi)
            if not isinstance(candidate['correct'], bool) or candidate['correct'] != (smi == target):
                raise ValueError('Candidate correctness does not match its recorded target')
            features = np.asarray(candidate['features'], dtype=float)
            if features.shape != (len(FEATURES),) or not np.isfinite(features).all():
                raise ValueError('Invalid candidate feature schema or nonfinite features')
        if total > samples:
            raise ValueError('Candidate counts exceed generated sample count')
        baseline_correct = bool(row['candidates'] and row['candidates'][0]['correct'])
        if row.get('top1_correct') is not baseline_correct:
            raise ValueError('Baseline correctness disagrees with the first candidate')
    component_groups = defaultdict(set)
    for row, group in zip(canonical_rows, components(canonical_rows)):
        component_groups[group].add(row['split_group'])
    if any(len(values) != 1 for values in component_groups.values()):
        raise ValueError('Connected pathway/reactant rows have different split_group values')
    return rows


def load_records(path, expected_split):
    data = Path(path).read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    return validate_records(rows, expected_split), hashlib.sha256(data).hexdigest()


def read_holdout_manifest(path):
    if path is None:
        return None
    manifest = json.loads(Path(path).read_text())
    required = {'row_id', 'source_row_id', 'source_split', 'split_group', 'sequence_id',
                'source_line_sha256', 'reactant_key'}
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('Holdout manifest must be the nonempty list of frozen row metadata')
    ids, source_ids = set(), set()
    for entry in manifest:
        if not isinstance(entry, dict) or not required <= entry.keys() or entry['source_split'] != 'test':
            raise ValueError('Holdout metadata is missing required identity/group/hash fields')
        if str(entry['row_id']) in ids or entry['source_row_id'] in source_ids:
            raise ValueError('Duplicate row identity in holdout manifest')
        ids.add(str(entry['row_id'])); source_ids.add(entry['source_row_id'])
    return manifest


def assert_manifest_disjoint(signature, manifest, context):
    if manifest is None:
        return
    expected = {
        'reactants': {entry['reactant_key'] for entry in manifest},
        'sequences': {entry['sequence_id'] for entry in manifest} - GENERIC_SEQUENCE_IDS,
        'groups': {entry['split_group'] for entry in manifest}}
    actual = dict(signature, reactants=[hashlib.sha256(value.encode()).hexdigest() for value in signature['reactants']])
    assert_disjoint(actual, expected, context)


def assert_manifest_membership(rows, manifest):
    if manifest is None:
        return
    expected = {str(entry['row_id']): entry for entry in manifest}
    if len(rows) != len(manifest) or {str(row['row_id']) for row in rows} != set(expected):
        raise ValueError('Evaluation membership differs from the precommitted holdout manifest')
    for row in rows:
        entry = expected[str(row['row_id'])]
        for field in ('source_row_id', 'source_split', 'split_group', 'sequence_id', 'source_line_sha256'):
            if row.get(field) != entry[field]:
                raise ValueError(f'Evaluation {field} differs from precommitted holdout metadata')
        digest = hashlib.sha256(canonical(row['reactant']).encode()).hexdigest()
        if digest != entry['reactant_key']:
            raise ValueError('Evaluation reactant differs from precommitted holdout metadata')


def implementation_signature():
    return {name: sha256(Path(__file__).with_name(name))
            for name in ('pilot_reranker.py', 'rerank_analysis.py')}


def runtime_versions():
    return {'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__}


def frequency_scores(row, k):
    return np.array([math.log(c['count']/row['sample_count']) for c in row['candidates'][:k]])


def stable_choice(scores, margin=0.0):
    """Keep baseline unless a candidate exceeds it by the prespecified margin."""
    if len(scores) == 0:
        return None
    winner = int(np.argmax(scores))
    return winner if scores[winner]-scores[0] > margin+1e-12 else 0


def frequency_choices(rows):
    return [0 if row['candidates'] else None for row in rows]


def learned_choices(rows, model, config, k, cached_scores=None):
    if config['alpha'] == 0.0 or model is None:
        return frequency_choices(rows)
    scores = cached_scores if cached_scores is not None else [
        model.decision_function(np.array([c['features'] for c in row['candidates'][:k]]))
        if row['candidates'] else np.array([]) for row in rows]
    return [stable_choice((1-config['alpha'])*frequency_scores(row, k)+config['alpha']*value,
                          config['margin']) for row, value in zip(rows, scores)]


def load_energies(path, rows):
    """Absent, failed and unsupported scores remain explicit; never impute energy."""
    if path is None:
        return {}, None, None
    raw = Path(path).read_bytes()
    candidate_keys = {(str(row['row_id']), canonical(c['smiles']))
                      for row in rows for c in row['candidates']}
    scores, methods, solvents = {}, set(), set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key = (str(entry['row_id']), canonical(entry['smiles']))
        if key not in candidate_keys:
            raise ValueError('Energy table includes a row/candidate outside its declared dataset')
        if key in scores:
            raise ValueError('Duplicate row/candidate energy entry')
        if entry.get('status') == 'ok' and not math.isfinite(float(entry['energy_hartree'])):
            raise ValueError('Nonfinite physical energy')
        if not entry.get('method') or not entry.get('solvent'):
            raise ValueError('Energy records must declare method and solvent')
        methods.add(entry['method']); solvents.add(entry['solvent'])
        scores[key] = entry
    if len(methods) != 1 or len(solvents) != 1:
        raise ValueError('Energy table must use one declared method and solvent')
    protocol = {'method': next(iter(methods)), 'solvent': next(iter(solvents)), 'units': 'hartree'}
    return scores, hashlib.sha256(raw).hexdigest(), protocol


def relative_energies(row, scores, k):
    values = [scores.get((str(row['row_id']), canonical(c['smiles']))) for c in row['candidates'][:k]]
    if not values or not all(v and v.get('status') == 'ok' for v in values):
        return None
    values = np.array([float(v['energy_hartree']) for v in values])
    return (values-values.min())*HARTREE_TO_KCAL_MOL


def energy_choices(rows, scores, k, weight=None, margin=0.0):
    """None selects minimum energy; zero weight keeps frequency. Missing -> baseline."""
    choices, covered = [], []
    for row in rows:
        rel = relative_energies(row, scores, k)
        covered.append(rel is not None)
        if rel is None or weight == 0.0:
            choices.append(0 if row['candidates'] else None)
        elif weight is None:
            choices.append(stable_choice(-rel))
        else:
            choices.append(stable_choice(frequency_scores(row, k)-weight*rel, margin))
    return choices, covered


def outcome_rows(rows, choices, k, covered=None):
    if len(rows) != len(choices):
        raise ValueError('One choice is required for every evaluation row')
    output = []
    for i, (row, choice) in enumerate(zip(rows, choices)):
        candidates = row['candidates'][:k]
        baseline = candidates[0] if candidates else None
        if choice is not None and not 0 <= choice < len(candidates):
            raise ValueError('Winner lies outside the fixed candidate set')
        winner = candidates[choice] if choice is not None else None
        before, after = bool(baseline and baseline['correct']), bool(winner and winner['correct'])
        changed = bool(baseline and winner and baseline['smiles'] != winner['smiles'])
        transition = ('corrected' if not before and after else 'harmed' if before and not after
                      else 'unchanged_correct' if before else 'unchanged_wrong')
        output.append({'row_id': row['row_id'], 'split_group': row['split_group'],
                       'baseline_smiles': baseline['smiles'] if baseline else None,
                       'selected_smiles': winner['smiles'] if winner else None,
                       'selected_rank': choice+1 if choice is not None else None,
                       'baseline_correct': before, 'correct': after, 'changed': changed,
                       'transition': transition, 'changed_wrong_to_wrong': changed and not before and not after,
                       'target_in_top_k': any(c['correct'] for c in candidates),
                       'covered': bool(covered[i]) if covered is not None else bool(candidates)})
    return output


def summarize(outcomes):
    n = len(outcomes)
    counts = Counter(row['transition'] for row in outcomes)
    before, correct = (sum(row[field] for row in outcomes) for field in ('baseline_correct', 'correct'))
    return {'rows': n, 'baseline_correct': before, 'correct': correct,
            'corrected': counts['corrected'], 'harmed': counts['harmed'],
            'unchanged_correct': counts['unchanged_correct'], 'unchanged_wrong': counts['unchanged_wrong'],
            'changed_wrong_to_wrong': sum(row['changed_wrong_to_wrong'] for row in outcomes),
            'changed': sum(row['changed'] for row in outcomes),
            'covered_rows': sum(row['covered'] for row in outcomes),
            'oracle_top_k_correct': sum(row['target_in_top_k'] for row in outcomes),
            'target_absent_from_top_k': sum(not row['target_in_top_k'] for row in outcomes),
            'accuracy': correct/n if n else None, 'baseline_accuracy': before/n if n else None,
            'net_correct': correct-before, 'net_accuracy_delta': (correct-before)/n if n else None}


def tuning_key(summary):
    return summary['correct'], -summary['changed'], -summary['harmed']


def cluster_bootstrap(outcomes, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Sample whole groups, then compute the row-weighted paired accuracy change."""
    if not outcomes:
        raise ValueError('Cannot bootstrap an empty cohort')
    groups = defaultdict(lambda: [0, 0])
    for row in outcomes:
        groups[row['split_group']][0] += int(row['correct'])-int(row['baseline_correct'])
        groups[row['split_group']][1] += 1
    values = np.array([groups[group] for group in sorted(groups)], dtype=float)
    rng, estimates = np.random.default_rng(seed), np.empty(resamples)
    for start in range(0, resamples, 256):
        n = min(256, resamples-start)
        drawn = values[rng.integers(0, len(values), size=(n, len(values)))].sum(axis=1)
        estimates[start:start+n] = drawn[:, 0]/drawn[:, 1]
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {'method': 'paired percentile cluster bootstrap; uniform group sampling, row-weighted ratio',
            'group_count': len(values), 'resamples': resamples, 'seed': seed, 'confidence_level': .95,
            'lower': float(lo), 'upper': float(hi),
            'point_estimate': float(values[:, 0].sum()/values[:, 1].sum())}


def credible_benefit(summary, uncertainty):
    discordant, reasons = summary['corrected']+summary['harmed'], []
    if summary['net_accuracy_delta'] < .01:
        reasons.append('net improvement below 1 percentage point')
    if uncertainty['lower'] <= 0:
        reasons.append('95% paired group-bootstrap lower bound is not positive')
    if uncertainty['group_count'] < 30:
        reasons.append('fewer than 30 evaluation groups')
    if discordant < 20:
        reasons.append('fewer than 20 correctness-discordant rows')
    status = ('negative' if summary['net_accuracy_delta'] < 0 else
              'tiny' if uncertainty['group_count'] < 30 or discordant < 20 else
              'inconclusive' if reasons else 'credible_benefit')
    return {'status': status, 'passed': not reasons, 'reasons': reasons,
            'correctness_discordant_rows': discordant,
            'benchmark_action': 'eligible_to_resume_full_benchmark' if not reasons else 'keep_full_benchmark_paused'}


def fit_pairwise(rows, k):
    x, y, weights, pair_rows, paired_groups = [], [], [], 0, set()
    for row in rows:
        candidates = row['candidates'][:k]
        correct = next((c for c in candidates if c['correct']), None)
        wrong = [c for c in candidates if not c['correct']]
        if correct is None or not wrong:
            continue
        pair_rows += 1
        paired_groups.add(row['split_group'])
        for candidate in wrong:
            difference = np.array(correct['features'])-np.array(candidate['features'])
            x.extend([difference, -difference]); y.extend([1, 0])
            weights.extend([.5/len(wrong)]*2)
    info = {'rows': len(rows), 'groups': len({row['split_group'] for row in rows}),
            'rows_with_training_pairs': pair_rows, 'groups_with_training_pairs': len(paired_groups),
            'candidate_pairs': len(x)//2,
            'row_weighting': 'one total sample weight per row, equally over symmetric candidate pairs'}
    if len(x) < 40 or len(paired_groups) < 5:
        return {}, dict(info, status='insufficient_training_pairs_frequency_fallback')
    models = {}
    for c in REGULARIZATION:
        model = make_pipeline(StandardScaler(with_mean=False),
                              LogisticRegression(C=c, fit_intercept=False, max_iter=1000, random_state=42))
        model.fit(np.array(x), np.array(y), logisticregression__sample_weight=np.array(weights))
        models[c] = model
    return models, dict(info, status='fitted')


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Refusing to overwrite a fitted model or evaluation; use a fresh output directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_tune(args):
    if args.k < 1:
        raise ValueError('K must be positive')
    fit, fit_hash = load_records(args.fit, 'train')
    tune, tune_hash = load_records(args.tune, 'val')
    fit_sig, tune_sig = source_signature(fit), source_signature(tune)
    assert_disjoint(fit_sig, tune_sig, 'Training/validation')
    holdout_manifest = read_holdout_manifest(args.holdout_manifest)
    assert_manifest_disjoint(fit_sig, holdout_manifest, 'Training/committed test')
    assert_manifest_disjoint(tune_sig, holdout_manifest, 'Validation/committed test')
    if args.fit_energies and not args.tune_energies:
        raise ValueError('Fit energies alone cannot tune an energy weight; provide validation energies')
    _, fit_energy_hash, fit_energy_protocol = load_energies(args.fit_energies, fit)
    energies, tune_energy_hash, energy_protocol = load_energies(args.tune_energies, tune)
    if fit_energy_protocol is not None and fit_energy_protocol != energy_protocol:
        raise ValueError('Training and validation energy protocols differ')
    baseline = summarize(outcome_rows(tune, frequency_choices(tune), args.k))
    selected = {'C': None, 'alpha': 0., 'margin': 0.}
    selected_model, selected_summary = None, baseline
    tuning = [{'method': 'frequency', 'config': selected, 'summary': baseline}]
    models, training_info = fit_pairwise(fit, args.k)
    for c, model in models.items():
        scores = [model.decision_function(np.array([cand['features'] for cand in row['candidates'][:args.k]]))
                  if row['candidates'] else np.array([]) for row in tune]
        for alpha in ALPHAS:
            for margin in MARGINS:
                config = {'C': c, 'alpha': alpha, 'margin': margin}
                summary = summarize(outcome_rows(tune, learned_choices(tune, model, config, args.k, scores), args.k))
                tuning.append({'method': 'learned', 'config': config, 'summary': summary})
                if tuning_key(summary) > tuning_key(selected_summary):
                    selected, selected_model, selected_summary = config, model, summary
    hybrid, hybrid_summary = {'weight_per_kcal_mol': 0., 'margin': 0.}, baseline
    if args.tune_energies:
        for weight in ENERGY_WEIGHTS:
            for margin in MARGINS:
                choices, covered = energy_choices(tune, energies, args.k, weight, margin)
                summary = summarize(outcome_rows(tune, choices, args.k, covered))
                config = {'weight_per_kcal_mol': weight, 'margin': margin}
                tuning.append({'method': 'hybrid', 'config': config, 'summary': summary})
                if tuning_key(summary) > tuning_key(hybrid_summary):
                    hybrid, hybrid_summary = config, summary
    validation = {'frequency': baseline, 'learned': selected_summary}
    if args.tune_energies:
        validation['hybrid'] = hybrid_summary
    primary = max(validation, key=lambda name: tuning_key(validation[name]))
    protocol = {
        'primary_k': args.k, 'frequency_policy': 'descending count, original stable order for ties',
        'learned_score': '(1-alpha)*log(count/sample_count) + alpha*pairwise_logistic_decision_function',
        'hybrid_score': 'log(count/sample_count) - weight_per_kcal_mol * relative_energy_kcal_mol',
        'switch_policy': 'change only if best score exceeds baseline score by margin + 1e-12',
        'missing_energy_policy': 'all top-K candidates required; otherwise retain frequency winner',
        'selection': 'maximize validation correct rows, then minimize changed rows, then harmed rows; stable grid tie',
        'C_grid': REGULARIZATION, 'alpha_grid': [0., *ALPHAS], 'margin_grid': MARGINS,
        'energy_weight_grid': ENERGY_WEIGHTS if args.tune_energies else [],
        'bootstrap_resamples': BOOTSTRAP_RESAMPLES, 'bootstrap_seed': BOOTSTRAP_SEED,
        'credible_benefit_thresholds': {'minimum_net_accuracy_delta': .01, 'confidence_lower_bound_above': 0,
                                        'minimum_groups': 30, 'minimum_discordant_rows': 20},
        'denominator': 'all evaluation rows including baseline-correct, target-absent and no-valid-candidate rows',
        'primary_selection': 'frequency / learned / hybrid selected on validation before holdout evaluation',
        'physical_minimum_energy': 'secondary diagnostic; cannot independently trigger benchmark resumption',
    }
    inputs = {'fit_records_sha256': fit_hash, 'tune_records_sha256': tune_hash,
              'fit_energies_sha256': fit_energy_hash, 'tune_energies_sha256': tune_energy_hash,
              'holdout_manifest_sha256': sha256(args.holdout_manifest) if args.holdout_manifest else None}
    if sha256(args.fit) != fit_hash or sha256(args.tune) != tune_hash:
        raise ValueError('Development records changed while fitting')
    payload = {'version': VERSION, 'feature_names': FEATURES, 'k': args.k, 'model': selected_model,
               'learned_config': selected, 'hybrid_config': hybrid, 'energy_enabled': bool(args.tune_energies),
               'energy_protocol': energy_protocol, 'primary_method': primary, 'inputs': inputs,
               'fit_signature': fit_sig, 'tune_signature': tune_sig, 'analysis_protocol': protocol,
               'implementation_sha256': implementation_signature(), 'runtime_versions': runtime_versions()}
    out = fresh_directory(args.out)
    joblib.dump(payload, out/'frozen.joblib')
    receipt = {key: value for key, value in payload.items() if key != 'model'}
    receipt.update(model_sha256=sha256(out/'frozen.joblib'), frozen_at=time.time(),
                   training=training_info, validation=validation, status='frozen_before_test_evaluation')
    json_write(out/'frozen.json', receipt)
    json_write(out/'validation-grid.json', tuning)
    (out/'frozen.joblib').chmod(0o444); (out/'frozen.json').chmod(0o444)
    print(json.dumps({'model': str(out/'frozen.joblib'), 'primary_method': primary, 'training': training_info,
                      'validation': validation, 'learned_config': selected,
                      'hybrid_config': hybrid if args.tune_energies else None}, indent=2))
    return receipt


def evaluate(args):
    model_path = Path(args.model)
    receipt = json.loads((model_path.parent/'frozen.json').read_text())
    model_hash = sha256(model_path)
    if model_hash != receipt['model_sha256']:
        raise ValueError('Frozen model changed after validation selection')
    payload = joblib.load(model_path)
    if payload['version'] != VERSION or payload['feature_names'] != FEATURES:
        raise ValueError('Frozen model version or feature schema mismatch')
    if payload['implementation_sha256'] != implementation_signature():
        raise ValueError('Reranker/extraction implementation changed after model freezing')
    if payload['runtime_versions'] != runtime_versions():
        raise ValueError('Reranker runtime differs from the frozen environment')
    for field in ('inputs', 'analysis_protocol', 'primary_method', 'fit_signature', 'tune_signature'):
        # JSON normalizes the grid tuples to lists.
        if json.dumps(payload[field], sort_keys=True) != json.dumps(receipt[field], sort_keys=True):
            raise ValueError('Frozen receipt and serialized model disagree')
    expected_manifest_hash = payload['inputs']['holdout_manifest_sha256']
    if expected_manifest_hash and (not args.holdout_manifest or sha256(args.holdout_manifest) != expected_manifest_hash):
        raise ValueError('The precommitted holdout manifest is required and must be unchanged')
    rows, records_hash = load_records(args.records, 'test')
    assert_manifest_membership(rows, read_holdout_manifest(args.holdout_manifest))
    signature = source_signature(rows)
    assert_disjoint(payload['fit_signature'], signature, 'Training/test')
    assert_disjoint(payload['tune_signature'], signature, 'Validation/test')
    if payload['energy_enabled'] != bool(args.energies):
        raise ValueError('Evaluation energy availability must match the frozen protocol')
    energies, energy_hash, energy_protocol = load_energies(args.energies, rows)
    if energy_protocol != payload['energy_protocol']:
        raise ValueError('Evaluation energy method or solvent differs from the frozen protocol')
    binding = {'model_sha256': model_hash, 'records_sha256': records_hash, 'energies_sha256': energy_hash,
               'holdout_manifest_sha256': expected_manifest_hash, 'output': str(Path(args.out).resolve())}
    claim_path = model_path.parent/'evaluation-claim.json'
    if claim_path.exists():
        prior = json.loads(claim_path.read_text())
        if prior != binding:
            raise ValueError('Frozen model is bound to another evaluation; repeated holdout mutation is forbidden')
        result_path = Path(args.out)/'comparison.json'
        if not result_path.exists():
            raise ValueError('Prior evaluation is incomplete; inspect artifacts instead of selecting a new holdout')
        artifacts = json.loads((Path(args.out)/'artifact-receipt.json').read_text())
        if any(sha256(Path(args.out)/name) != digest for name, digest in artifacts.items()):
            raise ValueError('Saved evaluation artifacts changed; refusing an unverifiable replay')
        existing = json.loads(result_path.read_text())
        if existing['binding'] != binding:
            raise ValueError('Saved comparison is bound to different evaluation inputs')
        print(json.dumps(existing, indent=2))
        return existing
    out = fresh_directory(args.out)
    with claim_path.open('x') as claim:
        json.dump(binding, claim, indent=2, sort_keys=True); claim.write('\n')
    k = payload['k']
    choices = {'frequency': frequency_choices(rows),
               'learned': learned_choices(rows, payload['model'], payload['learned_config'], k)}
    covered = {}
    if payload['energy_enabled']:
        choices['physical'], covered['physical'] = energy_choices(rows, energies, k)
        hybrid = payload['hybrid_config']
        choices['hybrid'], covered['hybrid'] = energy_choices(rows, energies, k,
                                                               hybrid['weight_per_kcal_mol'], hybrid['margin'])
    methods, detailed = {}, {}
    for name, selected in choices.items():
        detailed[name] = outcome_rows(rows, selected, k, covered.get(name))
        summary = summarize(detailed[name])
        summary['paired_group_bootstrap'] = cluster_bootstrap(detailed[name])
        methods[name] = summary
    primary = payload['primary_method']
    gate = credible_benefit(methods[primary], methods[primary]['paired_group_bootstrap'])
    if primary == 'frequency':
        gate['reason_frequency_selected'] = 'No reranker beat retaining frequency under the validation selection rule'
    with (out/'decisions.jsonl').open('w') as stream:
        for i, row in enumerate(rows):
            record = {'row_id': row['row_id'], 'split_group': row['split_group'],
                      'methods': {name: detailed[name][i] for name in detailed}}
            stream.write(json.dumps(record, sort_keys=True)+'\n')
    report = {'version': VERSION, 'binding': binding, 'status': 'fixed_test_cohort_evaluated_once',
              'evaluated_at': time.time(), 'rows': len(rows), 'groups': len(signature['groups']), 'k': k,
              'primary_method_selected_on_validation': primary, 'methods': methods, 'benchmark_resume_gate': gate,
              'notes': ['All selected test rows are retained in every denominator.',
                        'Unchanged_wrong refers to correctness; changed wrong-to-wrong is counted separately.',
                        'Secondary-method intervals are descriptive and not multiple-comparison-adjusted.',
                        'The resume gate applies only to the method selected on validation.',
                        'The learned score is statistical compatibility, not physical energy.',
                        'Physical scores, when supplied, are endpoint energies, not barriers or Gibbs free energies.']}
    if sha256(args.records) != records_hash or sha256(model_path) != model_hash:
        raise ValueError('Frozen inputs changed during evaluation')
    json_write(out/'comparison.json', report)
    json_write(out/'artifact-receipt.json', {name: sha256(out/name) for name in ('comparison.json', 'decisions.jsonl')})
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit-tune')
    for flag in ('fit', 'tune', 'out'):
        fit.add_argument('--'+flag, type=Path, required=True)
    fit.add_argument('--k', type=int, default=5)
    fit.add_argument('--fit-energies', type=Path)
    fit.add_argument('--tune-energies', type=Path)
    fit.add_argument('--holdout-manifest', type=Path)
    fit.set_defaults(func=fit_tune)
    ev = sub.add_parser('evaluate')
    for flag in ('records', 'model', 'out'):
        ev.add_argument('--'+flag, type=Path, required=True)
    ev.add_argument('--energies', type=Path)
    ev.add_argument('--holdout-manifest', type=Path)
    ev.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
