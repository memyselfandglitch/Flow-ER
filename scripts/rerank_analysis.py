#!/usr/bin/env python3
"""Extract FlowER ranking errors and evaluate held-out candidate rerankers.

Canonicalization follows eval_multiGPU.py (explicit H, no atom maps/stereo).
Sample counts are Monte Carlo frequencies, not exact model likelihoods.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog('rdApp.warning')
PARSER = Chem.SmilesParserParams()
PARSER.removeHs = False
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)


def properties(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    return {
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'charge': sum(charges), 'abs_charge': sum(abs(q) for q in charges),
        'charged_atoms': sum(q != 0 for q in charges),
        'fragments': len(Chem.GetMolFrags(mol)),
        'radicals': sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'rings': rdMolDescriptors.CalcNumRings(mol),
        'aromatic_atoms': sum(a.GetIsAromatic() for a in mol.GetAtoms()),
    }


def heavy_skeleton(smiles):
    """Diagnostic graph signature ignoring proton/charge/radical placement."""
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles, PARSER), sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


FEATURES = ['log_frequency', 'fraction_of_valid', 'reactant_similarity', 'no_op',
            'delta_abs_charge', 'delta_charged_atoms', 'delta_fragments',
            'delta_radicals', 'delta_rings', 'delta_aromatic_atoms',
            'candidate_abs_charge', 'candidate_fragments', 'candidate_radicals']


def analyze_row(gt_line, prediction_line, index, max_k):
    fields = gt_line.strip().split('|')
    reactant, target = map(canonical, fields[0].split('>>'))
    metrics_s, nonsym_s, candidates_s = prediction_line.strip().split('|', 2)
    metrics = ast.literal_eval(metrics_s)
    raw = ast.literal_eval(candidates_s)
    if len(metrics) != 5 or any(not isinstance(v, int) or v < 0 for v in metrics):
        raise ValueError(f'Invalid tallies at row {index}')
    counts = {}
    for smi, count, conserved in raw:
        if not conserved or not isinstance(count, int) or count <= 0:
            raise ValueError(f'Unexpected candidate at row {index}')
        smi = canonical(smi)
        counts[smi] = counts.get(smi, 0) + count
    if sum(counts.values()) != metrics[0] + metrics[1]:
        raise ValueError(f'Candidate/tally mismatch at row {index}')
    if counts.get(target, 0) != metrics[0]:
        raise ValueError(f'Ground-truth alignment/tally mismatch at row {index}')
    ranked = sorted(counts, key=counts.get, reverse=True)  # stable upstream order
    target_rank = ranked.index(target) + 1 if target in ranked else None
    target_count = counts.get(target, 0)
    rank_best = 1 + sum(n > target_count for n in counts.values()) if target_rank else None
    rank_worst = sum(n >= target_count for n in counts.values()) if target_rank else None
    rp = properties(reactant)
    rfp = FP.GetFingerprint(Chem.MolFromSmiles(reactant, PARSER))
    sample_count = sum(metrics)
    candidates = []
    for rank, smi in enumerate(ranked[:max_k], 1):
        cp = properties(smi)
        similarity = DataStructs.TanimotoSimilarity(rfp, FP.GetFingerprint(Chem.MolFromSmiles(smi, PARSER)))
        feature = [math.log((counts[smi] + 0.5) / (sample_count + 1)),
                   counts[smi] / max(1, sum(counts.values())), similarity, float(smi == reactant)]
        feature += [cp[k] - rp[k] for k in ['abs_charge', 'charged_atoms', 'fragments', 'radicals', 'rings', 'aromatic_atoms']]
        feature += [cp[k] for k in ['abs_charge', 'fragments', 'radicals']]
        candidates.append({'rank': rank, 'smiles': smi, 'count': counts[smi],
                           'correct': smi == target, 'no_op': smi == reactant,
                           'formula_matches_reactant': cp['formula'] == rp['formula'],
                           'properties': cp, 'features': feature})
    top = candidates[0] if candidates else None
    sequence = fields[-1] if len(fields) > 1 else ''
    return {'row_id': index, 'sequence_id': sequence,
            'reaction': fields[0], 'reactant': reactant, 'target': target,
            'reactant_properties': rp, 'sample_count': sample_count,
            'valid_conserved_samples': sum(counts.values()), 'metrics': metrics,
            'target_rank': target_rank, 'target_rank_best_tie': rank_best,
            'target_rank_worst_tie': rank_worst,
            'top1_correct': target_rank == 1,
            'target_tied_for_first': bool(top and target_rank and target_count == top['count']),
            'top1_target_count_gap': top['count'] - target_count if top and target_rank else None,
            'candidates': candidates}


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def extract(args):
    log = args.inference_log.read_text(errors='replace')
    skipped = re.findall(r'([\d.]+)% data is unparseable', log)
    if not skipped or any(float(x) != 0 for x in skipped):
        raise ValueError('Cannot prove row alignment: require an inference log with 0.0% data is unparseable.')
    args.out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    prefix_hash = hashlib.sha256()
    examples = []
    targets_by_reactant = {}
    k_values = [3, 5, 10]
    total_gt = sum(1 for _ in args.gt.open())
    with args.gt.open() as gt, args.pred.open('rb') as pred, \
            (args.out/'records.jsonl').open('w') as records, \
            (args.out/'ranking_errors.jsonl').open('w') as errors:
        for index, raw in enumerate(pred):
            if not raw.endswith(b'\n'):
                stats['ignored_incomplete_final_line'] += 1
                break
            target = next(gt, None)
            if target is None:
                raise ValueError('More predictions than ground-truth rows')
            row = analyze_row(target, raw.decode(), index, max(k_values))
            prefix_hash.update(raw)
            records.write(json.dumps(row) + '\n')
            stats['rows'] += 1
            targets_by_reactant.setdefault(row['reactant'], Counter())[row['target']] += 1
            stats['top1_correct'] += row['top1_correct']
            stats['no_valid_candidates'] += not row['candidates']
            rank = row['target_rank']
            for k in k_values:
                hit = rank is not None and rank <= k
                stats[f'top{k}_correct'] += hit
                stats[f'top{k}_correct_top1_wrong'] += hit and not row['top1_correct']
                stats[f'top{k}_strict_frequency_errors'] += hit and not row['top1_correct'] and not row['target_tied_for_first']
            if rank is not None and 1 < rank <= max(k_values):
                errors.write(json.dumps(row) + '\n')
                stats['errors_tied_for_first'] += row['target_tied_for_first']
                stats['errors_top1_no_op'] += row['candidates'][0]['no_op']
                if len(examples) < 12:
                    examples.append(row)
    n = stats['rows']
    # Diagnose alternatives using only labels actually observed in this snapshot.
    error_path = args.out/'ranking_errors.jsonl'
    temp = args.out/'ranking_errors.tmp'
    with error_path.open() as source, temp.open('w') as output:
        for line in source:
            row = json.loads(line)
            known = targets_by_reactant[row['reactant']]
            row['top1_is_other_recorded_target_for_same_reactant'] = row['candidates'][0]['smiles'] in known
            row['same_heavy_atom_skeleton_top1_vs_target'] = heavy_skeleton(row['candidates'][0]['smiles']) == heavy_skeleton(row['target'])
            stats['errors_top1_is_other_observed_target'] += row['top1_is_other_recorded_target_for_same_reactant']
            stats['errors_same_heavy_atom_skeleton'] += row['same_heavy_atom_skeleton_top1_vs_target']
            output.write(json.dumps(row)+'\n')
    temp.replace(error_path)
    summary = {'status': 'complete' if n == total_gt else 'provisional_prefix',
               'ground_truth_rows': total_gt, 'predictions_analyzed': n,
               'prediction_prefix_sha256': prefix_hash.hexdigest(),
               'ground_truth_file': str(args.gt.resolve()), 'prediction_file': str(args.pred.resolve()),
               'ranking_policy': 'descending frequency; original stable order for ties; tie bounds also recorded',
               'canonicalization': 'explicit hydrogens; atom maps and stereochemistry ignored, as in FlowER evaluation',
               'feature_names': FEATURES, 'counts': dict(stats),
               'distinct_reactants':len(targets_by_reactant),
               'reactants_with_multiple_recorded_targets':sum(len(v)>1 for v in targets_by_reactant.values()),
               'deterministic_reactant_only_top1_ceiling_on_this_snapshot':sum(max(v.values()) for v in targets_by_reactant.values())/n if n else None,
               'accuracy': {f'top{k}': stats[f'top{k}_correct']/n if n else None for k in [1, 3, 5, 10]},
               'caution': 'Ordered prefix is not representative; extraction is diagnostic, not evidence of reranker efficacy.'}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.out/'examples.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['row_id', 'sequence_id', 'target_rank', 'top1_count', 'target_count', 'tied_for_first', 'top1_no_op', 'reactant', 'top1', 'target'])
        writer.writeheader()
        for row in examples:
            correct = next(c for c in row['candidates'] if c['correct'])
            writer.writerow({'row_id':row['row_id'], 'sequence_id':row['sequence_id'], 'target_rank':row['target_rank'],
                             'top1_count':row['candidates'][0]['count'], 'target_count':correct['count'],
                             'tied_for_first':row['target_tied_for_first'], 'top1_no_op':row['candidates'][0]['no_op'],
                             'reactant':row['reactant'], 'top1':row['candidates'][0]['smiles'], 'target':row['target']})
    print(json.dumps(summary, indent=2))


def components(records):
    """Join rows sharing a real pathway ID OR canonical reactant to avoid leakage."""
    parent = list(range(len(records)))
    seen = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(records):
        keys = [('reactant', row['reactant'])]
        if row['sequence_id'] not in ['', 'PM', 'RS', 'RC', 'PC']:
            keys.append(('sequence', row['sequence_id']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    labels = {}
    for i, row in enumerate(records):
        labels.setdefault(find(i), []).append(row['reactant'])
    hashes = {group: hashlib.sha256('\n'.join(sorted(set(values))).encode()).hexdigest() for group, values in labels.items()}
    return [hashes[find(i)] for i in range(len(records))]


def split(args):
    records = load_records(args.records)
    groups = components(records)
    args.out.mkdir(parents=True, exist_ok=True)
    files = {name:(args.out/f'{name}.jsonl').open('w') for name in ['fit','tune','holdout']}
    counts = Counter()
    group_counts = {name:set() for name in files}
    try:
        for row, group in zip(records, groups):
            bucket = int(group[:8], 16) % 10
            part = 'fit' if bucket < 6 else 'tune' if bucket < 8 else 'holdout'
            row['split_group'] = group
            files[part].write(json.dumps(row)+'\n')
            counts[part] += 1
            group_counts[part].add(group)
    finally:
        for f in files.values(): f.close()
    info = {'status':'exploratory_test_partition_not_official_test', 'seed_rule':'sha256 component modulo 10: 60/20/20',
            'rows':dict(counts),'groups':{k:len(v) for k,v in group_counts.items()},
            'warning':'Freeze this snapshot. New rows may merge components; do not reuse old fitted models after resplitting.'}
    (args.out/'split.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


def train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    rows = load_records(args.fit)
    x, y = [], []
    for row in rows:
        candidates = row['candidates'][:args.k]
        correct = next((c for c in candidates if c['correct']), None)
        if correct is None:
            continue
        for candidate in candidates:
            if candidate['correct']: continue
            difference = np.array(correct['features']) - np.array(candidate['features'])
            x.extend([difference, -difference]); y.extend([1, 0])
    if len(x) < 40 or len(set(components(rows))) < 5:
        raise ValueError('Too few independent training examples: need >=20 candidate pairs and >=5 groups.')
    # Scale without centering: a score difference must have zero intercept.
    model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000, random_state=42))
    model.fit(np.array(x), np.array(y))
    payload = {'model': model, 'features': FEATURES, 'k': args.k,
               'fit_sha256': hashlib.sha256(args.fit.read_bytes()).hexdigest(),
               'fit_reactants': set(row['reactant'] for row in rows),
               'fit_sequences': set(row['sequence_id'] for row in rows) - {'','PM','RS','RC','PC'}}
    args.model.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(payload,args.model)
    print(json.dumps({'training_rows':len(rows),'pairwise_examples':len(x)//2,'model':str(args.model)}))


def compare(args):
    import joblib
    rows = load_records(args.records)
    payload = joblib.load(args.model) if args.model else None
    if payload:
        if payload['features'] != FEATURES or payload['k'] != args.k:
            raise ValueError('Model feature schema or K mismatch')
        if any(row['reactant'] in payload['fit_reactants'] or row['sequence_id'] in payload['fit_sequences'] for row in rows):
            raise ValueError('Training/evaluation group leakage detected')
    energies = {}
    if args.energies:
        for entry in load_records(args.energies):
            energies[(entry['row_id'], entry['smiles'])] = entry
    stats = {name: Counter() for name in ['frequency','learned','physical','hybrid']}
    for row in rows:
        candidates = row['candidates'][:args.k]
        original = candidates[0] if candidates else None
        winners = {'frequency':original}
        learned = None
        if payload and candidates:
            scores = payload['model'].decision_function(np.array([c['features'] for c in candidates]))
            learned = candidates[int(np.argmax(scores))]
        if payload: winners['learned'] = learned
        if args.energies:
            values = [energies.get((row['row_id'], c['smiles'])) for c in candidates]
            # Complete candidate coverage only; otherwise abstain and keep frequency top-1.
            eligible = bool(values) and all(v and v['status']=='ok' for v in values)
            for name in ['physical','hybrid']:
                stats[name]['coverage_rows'] += eligible
            if eligible:
                e = np.array([v['energy_hartree'] for v in values])
                rel = (e - e.min()) * 627.509474
                winners['physical'] = candidates[int(np.argmin(e))]
                hybrid = np.array([math.log(c['count'] / row['sample_count']) for c in candidates]) - args.energy_weight * rel
                winners['hybrid'] = candidates[int(np.argmax(hybrid))]
            else:
                winners['physical'] = winners['hybrid'] = original
        for name, winner in winners.items():
            s = stats[name]
            before = bool(original and original['correct'])
            after = bool(winner and winner['correct'])
            s['rows'] += 1; s['correct'] += after
            s['rescued'] += not before and after
            s['harmed'] += before and not after
            s['changed'] += bool(original and winner and original['smiles'] != winner['smiles'])
    result = {'k':args.k,'rows':len(rows),'energy_weight_per_kcal_mol':args.energy_weight,
              'oracle_topk_correct':sum(any(c['correct'] for c in r['candidates'][:args.k]) for r in rows),
              'methods':{k:{**dict(v),'accuracy':v['correct']/v['rows'] if v['rows'] else None} for k,v in stats.items() if v['rows']},
              'notes':'Physical missing/unsupported rows fall back to frequency. Fit/tune/holdout must remain separate. Learned score is statistical compatibility, not physical energy.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    p = subs.add_parser('extract')
    for name in ['gt','pred','inference-log','out']:p.add_argument('--'+name,type=Path,required=True)
    p.set_defaults(func=extract)
    p = subs.add_parser('split')
    p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.set_defaults(func=split)
    p = subs.add_parser('train')
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--k',type=int,default=10);p.set_defaults(func=train)
    p = subs.add_parser('compare')
    p.add_argument('--records',type=Path,required=True);p.add_argument('--model',type=Path)
    p.add_argument('--energies',type=Path);p.add_argument('--k',type=int,default=10)
    p.add_argument('--energy-weight',type=float,default=0.0,help='Tune on development data only; zero retains frequency ranking.')
    p.add_argument('--out',type=Path,required=True);p.set_defaults(func=compare)
    args=parser.parse_args();args.func(args)


if __name__ == '__main__':
    main()
