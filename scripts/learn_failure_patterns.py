#!/usr/bin/env python3
"""Exploratory learning from a diagnosed test prefix; never an untouched assessment.

Official source_split stays 'test'. Frozen groups define fit/tune/assessment
roles. Only input/candidate support and confidence control inference fallback.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from rerank_analysis import FEATURES, PARSER, canonical
from pilot_reranker import cluster_bootstrap, outcome_rows, summarize, stable_choice, tuning_key

STATUS = 'exploratory_not_untouched'
VERSION = 'diagnostic-prefix-learning-v1'
GENERIC = {'', 'PM', 'RS', 'RC', 'PC'}
ROLES = ('fit', 'tune', 'assessment')
CS = (.01, .1, 1.)
ALPHAS = (.1, .25, .5, .75, 1.)
MARGINS = (0., .1, .25, .5, 1.)
MAX_TOP1_FRACTIONS = (.5, .7, 1.)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Refusing to overwrite frozen development or assessment artifacts')
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=50000)
def input_key(smiles):
    try:
        with rdBase.BlockLogs():
            return 'canonical:'+canonical(smiles)
    except Exception:
        return 'unparseable_raw:'+smiles


def signature(rows):
    return {'groups': sorted({row['split_group'] for row in rows}),
            'reactants': sorted({input_key(row['reactant']) for row in rows}),
            'sequences': sorted({row['sequence_id'] for row in rows}-GENERIC)}


def assert_disjoint(left, right):
    for field in ('groups', 'reactants', 'sequences'):
        if set(left[field]) & set(right[field]):
            raise ValueError(f'Development-role leakage through shared {field}')


def validate_records(rows):
    if not rows:
        raise ValueError('No diagnostic records supplied')
    ids, seen_keys = set(), {}
    for row in rows:
        if row.get('source_split') != 'test':
            raise ValueError("Official source_split must stay 'test'; never relabel diagnostic rows as train/val")
        identifier = str(row['row_id'])
        if identifier in ids:
            raise ValueError('Duplicate row_id')
        ids.add(identifier)
        if not isinstance(row.get('split_group'), str) or not row['split_group']:
            raise ValueError('A frozen split_group is required')
        sequence = row['sequence_id']
        if sequence not in GENERIC and not sequence.startswith('test:'):
            raise ValueError('Real sequence IDs must be namespaced with test:')
        if not isinstance(row.get('rerank_eligible'), bool):
            raise ValueError('rerank_eligible must be a label-independent boolean support flag')
        samples = row['sample_count']
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError('Invalid sample_count')
        previous, total, correct = float('inf'), 0, 0
        for rank, candidate in enumerate(row['candidates'], 1):
            count = candidate['count']
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count > previous:
                raise ValueError('Candidates must retain descending positive integer counts')
            if candidate['rank'] != rank or not isinstance(candidate['smiles'], str):
                raise ValueError('Saved candidate order and raw strings must remain intact')
            if not isinstance(candidate['correct'], bool):
                raise ValueError('Candidate correctness must be a boolean label')
            previous = count; total += count; correct += candidate['correct']
        if total > samples or correct > 1:
            raise ValueError('Inconsistent candidate counts or target labels')
        baseline = bool(row['candidates'] and row['candidates'][0]['correct'])
        if row['top1_correct'] is not baseline:
            raise ValueError('Baseline label disagrees with the saved first candidate')
        keys = [('reactant', input_key(row['reactant']))]
        if sequence not in GENERIC:
            keys.append(('sequence', sequence))
        for key in keys:
            if key in seen_keys and seen_keys[key] != row['split_group']:
                raise ValueError('Related input/pathway rows have inconsistent frozen split_group values')
            seen_keys[key] = row['split_group']
    return rows


def load_records(path):
    checksum, rows = hashlib.sha256(), []
    with Path(path).open('rb') as stream:
        for line in stream:
            checksum.update(line)
            if line.strip():
                rows.append(json.loads(line))
    return validate_records(rows), checksum.hexdigest()


def make_splits(rows, source_hash, seed=20261028, manifest_path=None):
    groups = {row['split_group'] for row in rows}
    if manifest_path:
        supplied = json.loads(Path(manifest_path).read_text())
        if supplied.get('source_records_sha256', source_hash) != source_hash:
            raise ValueError('Split manifest describes different source records')
        assignments = supplied['group_assignments']
        if set(assignments) != groups or any(value not in ROLES for value in assignments.values()):
            raise ValueError('Split manifest must assign every source group exactly once')
        rule = 'supplied frozen group assignments'
    else:
        assignments = {}
        for group in sorted(groups):
            bucket = int(digest(f'{seed}:{group}'.encode())[:8], 16) % 10
            assignments[group] = 'fit' if bucket < 6 else 'tune' if bucket < 8 else 'assessment'
        rule = 'sha256(seed:split_group) first 8 hex modulo 10; 0-5 fit, 6-7 tune, 8-9 assessment'
    parts = {role: [row for row in rows if assignments[row['split_group']] == role] for role in ROLES}
    if any(not part for part in parts.values()):
        raise ValueError('Each development role needs at least one group')
    signatures = {role: signature(part) for role, part in parts.items()}
    for i, left in enumerate(ROLES):
        for right in ROLES[i+1:]:
            assert_disjoint(signatures[left], signatures[right])
    manifest = {'version': VERSION, 'status': STATUS, 'source_split': 'test',
                'source_records_sha256': source_hash, 'seed': seed, 'assignment_rule': rule,
                'group_assignments': assignments,
                'row_ids': {role: [row['row_id'] for row in part] for role, part in parts.items()},
                'counts': {role: {'rows': len(part), 'groups': len(signatures[role]['groups'])}
                           for role, part in parts.items()}}
    return parts, manifest, signatures


def transformation_bits(reactant, product):
    """Separate input-conditioned added/removed bits; do not subtract common input."""
    reactant, product = np.asarray(reactant, dtype=bool), np.asarray(product, dtype=bool)
    return np.concatenate((product & ~reactant, reactant & ~product)).astype(np.float64)


class FeatureBuilder:
    def __init__(self, k=10, fingerprint_bits=512):
        self.k, self.bits = k, fingerprint_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_bits)

    @lru_cache(maxsize=40000)
    def fingerprint(self, smiles):
        try:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles, PARSER)
            if mol is None:
                return None
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            bits = np.zeros(self.bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(self.generator.GetFingerprint(mol), bits)
            bits.setflags(write=False)
            return bits
        except Exception:
            return None

    def row_matrix(self, row, family):
        # Labels, target, target_rank, diagnostic flags and group/row IDs are not
        # read here. Failure to support ANY candidate keeps the exact baseline.
        if not row['rerank_eligible'] or not row['candidates']:
            return None
        candidates = row['candidates'][:self.k]
        if not all(candidate.get('parseable') is True for candidate in candidates):
            return None
        try:
            numerical = np.array([candidate['features'] for candidate in candidates], dtype=float)
        except (ValueError, TypeError, KeyError):
            return None
        if numerical.shape != (len(candidates), len(FEATURES)) or not np.isfinite(numerical).all():
            return None
        if input_key(row['reactant']).startswith('unparseable_raw:'):
            return None
        if family == 'simple':
            return sparse.csr_matrix(numerical)
        reactant = self.fingerprint(row['reactant'])
        products = [self.fingerprint(candidate['smiles']) for candidate in candidates]
        if reactant is None or any(product is None for product in products):
            return None
        changes = np.stack([transformation_bits(reactant, product) for product in products])
        return sparse.csr_matrix(np.hstack((numerical, changes)))

    def names(self, family):
        if family == 'simple':
            return list(FEATURES)
        return list(FEATURES)+[f'morgan_added_r2_bit_{i}' for i in range(self.bits)]+[
            f'morgan_removed_r2_bit_{i}' for i in range(self.bits)]


def baseline_choices(rows):
    return [0 if row['candidates'] else None for row in rows]


def choose(rows, scores, config, k):
    choices = []
    for row, learned in zip(rows, scores):
        candidates = row['candidates'][:k]
        confident = bool(candidates and candidates[0]['count']/row['sample_count'] > config['max_top1_fraction'])
        if learned is None or config['alpha'] == 0 or confident:
            choices.append(0 if candidates else None)
            continue
        frequency = np.log([candidate['count']/row['sample_count'] for candidate in candidates])
        values = (1-config['alpha'])*frequency+config['alpha']*learned
        choices.append(stable_choice(values, config['margin']))
    return choices


def model_scores(matrices, model):
    nonempty = [matrix for matrix in matrices if matrix is not None]
    if not nonempty:
        return [None]*len(matrices)
    flat = model.decision_function(sparse.vstack(nonempty, format='csr'))
    result, offset = [], 0
    for matrix in matrices:
        if matrix is None:
            result.append(None)
        else:
            length = matrix.shape[0]
            result.append(flat[offset:offset+length]); offset += length
    return result


def fit_models(rows, builder, family):
    differences, weights, paired_groups = [], [], set()
    eligible, pair_rows, skipped, no_negatives = 0, 0, 0, 0
    alternative_targets = {}
    if family == 'transform_fingerprint':
        # These labels come only from the fit role. Neither tuning nor assessment
        # targets are accepted by this function or consulted during inference.
        for row in rows:
            target = input_key(row['target'])
            if target.startswith('canonical:'):
                alternative_targets.setdefault(input_key(row['reactant']), set()).add(target)
    for row in rows:
        features = builder.row_matrix(row, family)
        if features is None:
            continue
        eligible += 1
        candidates = row['candidates'][:builder.k]
        correct = next((i for i, candidate in enumerate(candidates) if candidate['correct']), None)
        wrong = [i for i, candidate in enumerate(candidates) if not candidate['correct']]
        if correct is None or not wrong:
            continue
        if family == 'transform_fingerprint':
            known = alternative_targets.get(input_key(row['reactant']), set())
            retained = [i for i in wrong if input_key(candidates[i]['smiles']) not in known]
            skipped += len(wrong)-len(retained)
            wrong = retained
            if not wrong:
                no_negatives += 1
                continue
        pair_rows += 1; paired_groups.add(row['split_group'])
        for negative in wrong:
            difference = features[correct]-features[negative]
            differences.extend((difference, -difference))
            weights.extend((.5/len(wrong), .5/len(wrong)))
    info = {'rows': len(rows), 'eligible_rows': eligible, 'paired_rows': pair_rows,
            'paired_groups': len(paired_groups), 'candidate_pairs': len(differences)//2,
            'negative_policy': 'fit_observed_alternative_targets_are_not_negatives' if family == 'transform_fingerprint' else 'strict_recorded_target_negatives',
            'contradictory_negative_pairs_skipped': skipped,
            'rows_without_negatives_after_filter': no_negatives,
            'fit_reactants_with_multiple_observed_targets': sum(len(values)>1 for values in alternative_targets.values()),
            'weighting': 'one total sample weight per input row across symmetric candidate pairs'}
    if len(differences) < 40 or len(paired_groups) < 5:
        return {}, dict(info, status='insufficient_training_signal_frequency_fallback')
    x, y = sparse.vstack(differences, format='csr'), np.tile([1, 0], len(differences)//2)
    models, rejected = {}, []
    for c in CS:
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(
            C=c, fit_intercept=False, solver='liblinear', max_iter=2000, random_state=42))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y, logisticregression__sample_weight=np.array(weights))
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            rejected.append(c)
        else:
            models[c] = model
    return models, dict(info, status='fitted' if models else 'no_converged_model_frequency_fallback', rejected_C=rejected)


def tune_family(fit, tune, builder, family):
    models, training = fit_models(fit, builder, family)
    matrices = [builder.row_matrix(row, family) for row in tune]
    eligible = [matrix is not None for matrix in matrices]
    selected = {'C': None, 'alpha': 0., 'margin': 0., 'max_top1_fraction': 1.}
    best = summarize(outcome_rows(tune, baseline_choices(tune), builder.k, eligible))
    grid = [{'config': selected, 'summary': best}]
    for c, model in models.items():
        scores = model_scores(matrices, model)
        for alpha in ALPHAS:
            for margin in MARGINS:
                for confidence in MAX_TOP1_FRACTIONS:
                    config = {'C': c, 'alpha': alpha, 'margin': margin, 'max_top1_fraction': confidence}
                    selected_rows = choose(tune, scores, config, builder.k)
                    summary = summarize(outcome_rows(tune, selected_rows, builder.k, eligible))
                    grid.append({'config': config, 'summary': summary})
                    if tuning_key(summary) > tuning_key(best):
                        best, selected = summary, config
    # Keep a fitted candidate for weight inspection even if alpha=0 wins on
    # validation. Its actual scoring contribution is then exactly zero.
    model_c = selected['C'] if selected['C'] is not None else min(models, default=None)
    return {'model': models.get(model_c), 'fitted_C': model_c, 'config': selected, 'training': training,
            'validation': best, 'feature_names': builder.names(family)}, grid


def top_weights(payload, limit=20):
    if payload['model'] is None:
        return {'status': 'no_fitted_model', 'weights': []}
    scaler, logistic = payload['model'].steps[0][1], payload['model'].steps[1][1]
    standardized = logistic.coef_[0]
    coefficients = standardized/scaler.scale_
    indices = np.argsort(-np.abs(coefficients), kind='stable')[:limit]
    return {'status': 'active' if payload['config']['alpha'] else 'fitted_candidate_inactive_frequency_selected',
            'fitted_C': payload['fitted_C'], 'frequency_prior_weight': 1-payload['config']['alpha'],
            'switch_margin': payload['config']['margin'], 'max_top1_fraction': payload['config']['max_top1_fraction'],
            'weights': [{'feature': payload['feature_names'][i], 'coefficient_raw_units': float(coefficients[i]),
                         'coefficient_standardized_units': float(standardized[i]),
                         'active_coefficient': float(payload['config']['alpha']*coefficients[i])} for i in indices],
            'interpretation': 'Hashed Morgan bits can collide; a weight alone does not identify a unique chemical mechanism.'}


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ('learn_failure_patterns.py', 'pilot_reranker.py', 'rerank_analysis.py')}


def runtime_versions():
    return {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'rdkit': rdBase.rdkitVersion, 'joblib': joblib.__version__}


def fit_tune(args):
    rows, source_hash = load_records(args.records)
    parts, splits, signatures = make_splits(rows, source_hash, args.seed, args.split_manifest)
    builder = FeatureBuilder(args.k, args.fingerprint_bits)
    families = ['simple']+([] if args.no_fingerprints else ['transform_fingerprint'])
    learned, grid = {}, {}
    baseline = summarize(outcome_rows(parts['tune'], baseline_choices(parts['tune']), args.k))
    validation = {'frequency': baseline}
    for family in families:
        learned[family], grid[family] = tune_family(parts['fit'], parts['tune'], builder, family)
        validation[family] = learned[family]['validation']
    primary = max(validation, key=lambda family: tuning_key(validation[family]))
    protocol = {'k': args.k, 'fingerprint_bits': args.fingerprint_bits, 'fingerprint_radius': 2,
                'fingerprint_form': 'concatenated added and removed Morgan presence bits, conditioned on input',
                'transformation_negative_policy': 'skip candidates that are alternative observed targets for the same canonical input in FIT labels only',
                'comparison_scope': 'numeric strict-negative model versus fingerprint-plus-ambiguity-policy variant; not a fingerprint-only ablation',
                'numeric_features': FEATURES, 'families': families, 'C_grid': list(CS),
                'alpha_grid': [0., *ALPHAS], 'margin_grid': list(MARGINS),
                'max_top1_fraction_grid': list(MAX_TOP1_FRACTIONS),
                'selection': 'validation correct count, fewer changes, fewer harms; stable grid ties',
                'eligibility': 'input/candidate support only; never target presence or diagnostic outcome labels',
                'training_target_absent': 'excluded from pairwise fitting only',
                'assessment_target_absent': 'retained; no label-informed fallback; wrong-to-wrong changes counted',
                'source_split': 'test', 'status': STATUS, 'bootstrap_resamples': 10000,
                'bootstrap_seed': 20261028, 'benchmark_resume_allowed': False}
    if file_hash(args.records) != source_hash:
        raise ValueError('Source records changed during fitting')
    out = fresh_directory(args.out)
    write_json(out/'splits.json', splits)
    frozen = {'version': VERSION, 'status': STATUS, 'source_records_sha256': source_hash,
              'source_records': str(Path(args.records).resolve()), 'split_manifest_sha256': file_hash(out/'splits.json'),
              'signatures': signatures, 'protocol': protocol, 'models': learned,
              'validation_selected_method': primary, 'implementation_sha256': implementation_hashes(),
              'runtime_versions': runtime_versions(), 'frozen_at': time.time()}
    joblib.dump(frozen, out/'frozen.joblib')
    receipt = {key: value for key, value in frozen.items() if key != 'models'}
    receipt.update(model_sha256=file_hash(out/'frozen.joblib'), validation=validation,
                   training={family: fitted['training'] for family, fitted in learned.items()},
                   selected_configurations={family: fitted['config'] for family, fitted in learned.items()},
                   top_weights={family: top_weights(fitted) for family, fitted in learned.items()})
    write_json(out/'frozen.json', receipt)
    write_json(out/'validation-grid.json', grid)
    for name in ('frozen.joblib', 'frozen.json', 'splits.json'):
        (out/name).chmod(0o444)
    print(json.dumps({'status': STATUS, 'model': str(out/'frozen.joblib'), 'split_counts': splits['counts'],
                      'validation_selected_method': primary, 'validation': validation,
                      'benchmark_resume_allowed': False}, indent=2), flush=True)
    return receipt


def assess(args):
    model_path = Path(args.model)
    receipt = json.loads((model_path.parent/'frozen.json').read_text())
    if file_hash(model_path) != receipt['model_sha256']:
        raise ValueError('Frozen model changed after tuning')
    payload = joblib.load(model_path)
    if payload['implementation_sha256'] != implementation_hashes() or payload['runtime_versions'] != runtime_versions():
        raise ValueError('Implementation/runtime differs from the frozen development run')
    split_path = model_path.parent/'splits.json'
    if file_hash(split_path) != payload['split_manifest_sha256']:
        raise ValueError('Frozen group split changed')
    rows, source_hash = load_records(args.records)
    if source_hash != payload['source_records_sha256']:
        raise ValueError('Assessment source differs from the frozen diagnostic cohort')
    parts, splits, signatures = make_splits(rows, source_hash, manifest_path=split_path)
    for role in ROLES:
        if signatures[role] != payload['signatures'][role]:
            raise ValueError('Membership differs from the frozen development roles')
    binding = {'model_sha256': receipt['model_sha256'], 'source_records_sha256': source_hash,
               'split_manifest_sha256': payload['split_manifest_sha256'], 'output': str(Path(args.out).resolve())}
    claim = model_path.parent/'assessment-claim.json'
    if claim.exists():
        if json.loads(claim.read_text()) != binding:
            raise ValueError('Frozen model is already bound to another assessment')
        artifacts = json.loads((Path(args.out)/'artifact-receipt.json').read_text())
        if any(file_hash(Path(args.out)/name) != value for name, value in artifacts.items()):
            raise ValueError('Saved assessment artifacts changed')
        report = json.loads((Path(args.out)/'comparison.json').read_text())
        print(json.dumps(report, indent=2), flush=True)
        return report
    out = fresh_directory(args.out)
    with claim.open('x') as stream:
        json.dump(binding, stream, indent=2, sort_keys=True)
    assessment = parts['assessment']
    k = payload['protocol']['k']
    builder = FeatureBuilder(k, payload['protocol']['fingerprint_bits'])
    detailed = {'frequency': outcome_rows(assessment, baseline_choices(assessment), k)}
    for family, fitted in payload['models'].items():
        matrices = [builder.row_matrix(row, family) for row in assessment]
        scores = model_scores(matrices, fitted['model']) if fitted['model'] is not None else [None]*len(assessment)
        selected = choose(assessment, scores, fitted['config'], k)
        detailed[family] = outcome_rows(assessment, selected, k, [matrix is not None for matrix in matrices])
    methods = {}
    for name, outcomes in detailed.items():
        summary = summarize(outcomes)
        summary['net_accuracy_change_pp'] = 100*summary['net_accuracy_delta']
        summary['paired_group_bootstrap'] = cluster_bootstrap(outcomes)
        summary['eligible_coverage'] = summary['covered_rows']/summary['rows']
        methods[name] = summary
    with (out/'choices.jsonl').open('w') as stream:
        for i, row in enumerate(assessment):
            record = {'row_id': row['row_id'], 'source_row_id': row['source_row_id'], 'source_split': 'test',
                      'development_role': 'assessment', 'split_group': row['split_group'], 'status': STATUS,
                      'methods': {name: values[i] for name, values in detailed.items()}}
            stream.write(json.dumps(record, sort_keys=True)+'\n')
    primary = payload['validation_selected_method']
    point, interval = methods[primary]['net_accuracy_delta'], methods[primary]['paired_group_bootstrap']
    signal = ('negative' if point < 0 else 'exploratory_positive_signal' if interval['lower'] > 0 else 'inconclusive')
    report = {'version': VERSION, 'status': STATUS, 'source_split': 'test', 'binding': binding,
              'assessed_at': time.time(), 'rows': len(assessment), 'groups': len(signatures['assessment']['groups']),
              'k': k, 'methods': methods, 'validation_selected_method': primary, 'signal': signal,
              'benchmark_resume_allowed': False,
              'top_weights': {family: top_weights(fitted) for family, fitted in payload['models'].items()},
              'notes': ['Prefix labels were available for diagnosis; group separation does not make this assessment untouched.',
                        'Every assessment row remains in the denominator, including unsupported and target-absent cases.',
                        'Only candidate/input support controls fallback; scoring never consults ground-truth target presence.',
                        'Frequency ties and unsupported candidates retain their exact saved order and raw strings.',
                        'Secondary model intervals are descriptive and not adjusted for multiple comparisons.',
                        'A future frozen evaluation on the separately reserved untouched cohort is needed before benchmark resumption.']}
    if file_hash(args.records) != source_hash or file_hash(model_path) != receipt['model_sha256']:
        raise ValueError('Frozen inputs changed during assessment')
    write_json(out/'comparison.json', report)
    write_json(out/'artifact-receipt.json', {name: file_hash(out/name) for name in ('comparison.json', 'choices.jsonl')})
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit-tune')
    fit.add_argument('--records', type=Path, required=True)
    fit.add_argument('--out', type=Path, required=True)
    fit.add_argument('--k', type=int, default=10, choices=range(1, 11))
    fit.add_argument('--fingerprint-bits', type=int, default=512, choices=(512, 1024))
    fit.add_argument('--no-fingerprints', action='store_true')
    fit.add_argument('--seed', type=int, default=20261028)
    fit.add_argument('--split-manifest', type=Path)
    fit.set_defaults(func=fit_tune)
    evaluation = sub.add_parser('assess')
    for name in ('records', 'model', 'out'):
        evaluation.add_argument('--'+name, type=Path, required=True)
    evaluation.set_defaults(func=assess)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
