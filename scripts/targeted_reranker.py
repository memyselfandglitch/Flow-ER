#!/usr/bin/env python3
"""Frozen top-10 CPU pilot: official train -> val tuning -> disjoint audit.

All chemistry features are recomputed from input/candidate strings. Candidate
graphs are unmapped: descriptor contrasts never assert atom correspondence.
The primary method is predeclared learned_targeted, including frequency fallback.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from pilot_reranker import cluster_bootstrap, outcome_rows, summarize, stable_choice, tuning_key

VERSION = 'targeted-chemistry-pilot-v1'
K = 10
PRIMARY = 'learned_targeted'
METHODS = ('frequency', 'chemistry_rule', 'learned_global', PRIMARY)
GENERIC = {'', 'PM', 'RS', 'RC', 'PC'}
CS = (.01, .1, 1.)
ALPHAS = (.25, .5)
MARGINS = (0., .5, 1.)
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
PARSER = Chem.SmilesParserParams()
PARSER.removeHs = False
PARSER.sanitize = False
GLOBAL_NAMES = [
    'log_frequency', 'fraction_of_retained_top10', 'reactant_similarity', 'no_op',
    'delta_abs_charge', 'delta_charged_atoms', 'delta_fragments', 'delta_radicals',
    'delta_rings', 'delta_aromatic_atoms', 'candidate_abs_charge',
    'candidate_fragments', 'candidate_radicals']
GROUPS = (
    'neutral_amine_nh', 'neutral_amine_or_imine_acceptor', 'amide_nh',
    'aromatic_n_acceptor', 'aromatic_nhplus', 'nonaromatic_nhplus',
    'highcoord_nhplus', 'oxonium_h', 'carboxylate_o', 'alkoxide_o',
    'hydroxide_o', 'other_anionic_o', 'acid_oh', 'alcohol_oh', 'water_o',
    'ether_o', 'carbonyl_o', 'acyl_halide', 'ester_carbonyl',
    'amide_carbonyl', 'other_carbonyl', 'tetrahedral_hetero_carbon',
    'heteroaryl_halide', 'alkyl_halide', 'hcl', 'chloride')
FLAGS = ('carbon_neighbors_gt4', 'carbon_bond_valence_gt4',
         'neutral_n_valence_gt3', 'neutral_o_valence_gt2',
         'carbon_charge_ge2', 'aluminum_charge_le_minus2', 'carbon_aluminum_charge_pair')
CONTEXT_NAMES = ('positive_h_donor_and_acceptor', 'hcl_and_alkoxide',
                 'acyl_halide_and_nh', 'ester_and_hydroxide',
                 'intramolecular_azaaryl_nh_ring5to8', 'intramolecular_azaaryl_nh_ring9to12')
PROGRESS_NAMES = ('same_heavy_connectivity', 'oxonium_deprotonation_context',
                  'nhplus_deprotonation_aromatization', 'hcl_alkoxide_neutralization',
                  'carbonyl_addition_context', 'ring_closure_context')
TARGETED_NAMES = (GLOBAL_NAMES + ['candidate_'+key for key in GROUPS]
                  + ['delta_'+key for key in GROUPS]
                  + ['candidate_'+key for key in FLAGS]
                  + ['delta_'+key for key in FLAGS]
                  + ['changed_x_input_'+key for key in CONTEXT_NAMES]
                  + ['no_op_x_input_'+key for key in CONTEXT_NAMES]
                  + list(PROGRESS_NAMES))
RULE_WEIGHTS = {'valence_flag': -2., 'carbon_aluminum_charge_pair': -2.,
                'oxonium_deprotonation_context': .25,
                'nhplus_deprotonation_aromatization': .5,
                'hcl_alkoxide_neutralization': .5}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Refusing to overwrite frozen pilot artifacts')
    path.mkdir(parents=True, exist_ok=True)
    return path


def total_h(atom):
    return atom.GetTotalNumHs(includeNeighbors=True)


def valence_flags(mol):
    """Structural flags, including explicit-H neighbors; not a full validity oracle."""
    flags = dict.fromkeys(FLAGS, 0.)
    for atom in mol.GetAtoms():
        z, q = atom.GetAtomicNum(), atom.GetFormalCharge()
        valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetNumExplicitHs()
        if z == 6:
            flags['carbon_neighbors_gt4'] += atom.GetDegree() > 4
            flags['carbon_bond_valence_gt4'] += valence > 4.001
            flags['carbon_charge_ge2'] += q >= 2
        if z == 7 and q == 0 and not atom.GetIsAromatic():
            flags['neutral_n_valence_gt3'] += valence > 3.001
        if z == 8 and q == 0:
            flags['neutral_o_valence_gt2'] += valence > 2.001
        if z == 13:
            flags['aluminum_charge_le_minus2'] += q <= -2
    flags['carbon_aluminum_charge_pair'] = float(
        flags['carbon_charge_ge2'] > 0 and flags['aluminum_charge_le_minus2'] > 0)
    return flags


def carbonyl_carbon(atom):
    return atom.GetAtomicNum() == 6 and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() == 8
        for b in atom.GetBonds())


def heavy_key(mol):
    """Canonical unlabeled heavy graph; equality does not supply an atom mapping."""
    graph, indices = Chem.RWMol(), {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            new = Chem.Atom(atom.GetAtomicNum())
            new.SetNoImplicit(True)
            indices[atom.GetIdx()] = graph.AddAtom(new)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in indices and b in indices:
            graph.AddBond(indices[a], indices[b], Chem.BondType.SINGLE)
    return Chem.MolToSmiles(graph.GetMol(), canonical=True, isomericSmiles=False)


@lru_cache(maxsize=50000)
def describe(smiles):
    """Input-only chemistry. Unsupported strings retain diagnostic flags but no scores."""
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles, PARSER)
            if mol is None or mol.GetNumAtoms() == 0:
                return {'supported': False, 'reason': 'parse_failure', 'flags': None}
            mol.UpdatePropertyCache(strict=False)
            flags = valence_flags(mol)
            failure = Chem.SanitizeMol(mol, catchErrors=True)
            if failure != Chem.SanitizeFlags.SANITIZE_NONE:
                return {'supported': False, 'reason': 'sanitize_failure', 'flags': flags}
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            counts = dict.fromkeys(GROUPS, 0.)
            amines, electrophiles = [], []
            aza_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()
                         if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and
                                mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
            for atom in mol.GetAtoms():
                z, q, h = atom.GetAtomicNum(), atom.GetFormalCharge(), total_h(atom)
                neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
                on_carbonyl = any(carbonyl_carbon(n) for n in neighbors)
                if z == 7:
                    if q == 0 and not atom.GetIsAromatic():
                        if on_carbonyl and h:
                            counts['amide_nh'] += 1
                        if not on_carbonyl:
                            counts['neutral_amine_or_imine_acceptor'] += 1
                            if h:
                                counts['neutral_amine_nh'] += 1
                                amines.append(atom.GetIdx())
                    counts['aromatic_n_acceptor'] += bool(atom.GetIsAromatic() and q == 0 and h == 0)
                    counts['aromatic_nhplus'] += bool(atom.GetIsAromatic() and q > 0 and h > 0)
                    counts['nonaromatic_nhplus'] += bool(not atom.GetIsAromatic() and q > 0 and h > 0)
                    counts['highcoord_nhplus'] += bool(q > 0 and h > 0 and len(neighbors) >= 3)
                if z == 8:
                    counts['oxonium_h'] += bool(q > 0 and h > 0)
                    counts['carbonyl_o'] += any(b.GetBondType() == Chem.BondType.DOUBLE and
                                                b.GetOtherAtom(atom).GetAtomicNum() == 6 for b in atom.GetBonds())
                    if q < 0:
                        category = ('carboxylate_o' if on_carbonyl else
                                    'hydroxide_o' if h and not neighbors else
                                    'alkoxide_o' if any(n.GetAtomicNum() == 6 for n in neighbors) else
                                    'other_anionic_o')
                        counts[category] += 1
                    if q == 0:
                        counts['water_o'] += bool(h == 2 and not neighbors)
                        counts['acid_oh'] += bool(h and on_carbonyl)
                        counts['alcohol_oh'] += bool(h and not on_carbonyl and
                                                   any(n.GetAtomicNum() == 6 for n in neighbors))
                        counts['ether_o'] += bool(h == 0 and len(neighbors) == 2 and not on_carbonyl and
                                                 all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()))
                if z == 6:
                    if carbonyl_carbon(atom):
                        single = [b.GetOtherAtom(atom).GetAtomicNum() for b in atom.GetBonds()
                                  if b.GetBondType() == Chem.BondType.SINGLE]
                        counts['acyl_halide'] += any(n in (9, 17, 35, 53) for n in single)
                        counts['ester_carbonyl'] += any(
                            b.GetBondType() == Chem.BondType.SINGLE and
                            b.GetOtherAtom(atom).GetAtomicNum() == 8 and
                            b.GetOtherAtom(atom).GetFormalCharge() == 0 and
                            total_h(b.GetOtherAtom(atom)) == 0 and
                            sum(n.GetAtomicNum() > 1 for n in b.GetOtherAtom(atom).GetNeighbors()) == 2
                            for b in atom.GetBonds())
                        counts['amide_carbonyl'] += 7 in single
                        counts['other_carbonyl'] += not any(n in (7, 8, 9, 17, 35, 53) for n in single)
                    counts['tetrahedral_hetero_carbon'] += bool(
                        not atom.GetIsAromatic() and sum(n.GetAtomicNum() in (7, 8) for n in neighbors) >= 2 and
                        all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()))
                    halogen = any(n.GetAtomicNum() in (9, 17, 35, 53) for n in neighbors)
                    if halogen and atom.GetIsAromatic() and any(atom.GetIdx() in r for r in aza_rings):
                        counts['heteroaryl_halide'] += 1
                        electrophiles.append(atom.GetIdx())
                    counts['alkyl_halide'] += bool(halogen and not atom.GetIsAromatic() and not carbonyl_carbon(atom))
                if z == 17:
                    counts['hcl'] += bool(q == 0 and h > 0)
                    counts['chloride'] += bool(q == -1 and not neighbors and h == 0)
            acceptors = sum(counts[key] for key in ('neutral_amine_or_imine_acceptor', 'aromatic_n_acceptor',
                                                   'carboxylate_o', 'alkoxide_o', 'hydroxide_o', 'other_anionic_o'))
            contexts = [float((counts['oxonium_h']+counts['nonaromatic_nhplus']) > 0 and acceptors > 0),
                        float(counts['hcl'] > 0 and counts['alkoxide_o']+counts['hydroxide_o'] > 0),
                        float(counts['acyl_halide'] > 0 and counts['neutral_amine_nh']+counts['amide_nh'] > 0),
                        float(counts['ester_carbonyl'] > 0 and counts['hydroxide_o'] > 0), 0., 0.]
            fragments = Chem.GetMolFrags(mol)
            component = {index: j for j, part in enumerate(fragments) for index in part}
            for amine in amines:
                for electrophile in electrophiles:
                    if component[amine] == component[electrophile]:
                        ring_size = len(Chem.GetShortestPath(mol, amine, electrophile))
                        contexts[4] = max(contexts[4], float(5 <= ring_size <= 8))
                        contexts[5] = max(contexts[5], float(9 <= ring_size <= 12))
            charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
            props = {'abs_charge': sum(abs(q) for q in charges), 'charged_atoms': sum(q != 0 for q in charges),
                     'fragments': len(fragments), 'radicals': sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
                     'rings': mol.GetRingInfo().NumRings(), 'aromatic_atoms': sum(a.GetIsAromatic() for a in mol.GetAtoms())}
            return {'supported': True, 'canonical': Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True),
                    'heavy_key': heavy_key(mol), 'counts': counts, 'flags': valence_flags(mol),
                    'context': np.asarray(contexts), 'properties': props, 'fp': FP.GetFingerprint(mol)}
    except Exception as exc:
        return {'supported': False, 'reason': 'descriptor_failure:'+type(exc).__name__, 'flags': None}


def input_key(smiles):
    result = describe(smiles)
    return result['canonical'] if result['supported'] else '!unparseable_raw:'+smiles


def signature(rows):
    return {'groups': sorted({row['split_group'] for row in rows}),
            'reactants': sorted({input_key(row['reactant']) for row in rows}),
            'sequences': sorted({row['sequence_id'] for row in rows}-GENERIC)}


def assert_disjoint(left, right, context='Partitions'):
    for key in ('groups', 'reactants', 'sequences'):
        if set(left[key]) & set(right[key]):
            raise ValueError(f'{context}: leakage through shared {key}')


def validate_records(rows, expected_split):
    if not rows:
        raise ValueError('No records supplied')
    identities, sources, connected = set(), set(), {}
    for row in rows:
        if row.get('source_split') != expected_split:
            raise ValueError(f'Expected source_split={expected_split}; diagnostic test-prefix fitting is forbidden')
        for key, seen in (('row_id', identities), ('source_row_id', sources)):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) in seen:
                raise ValueError(f'{key} must be present and unique')
            seen.add(str(value))
        if not isinstance(row.get('split_group'), str) or not row['split_group']:
            raise ValueError('A frozen split_group is required')
        sequence = row.get('sequence_id')
        if not isinstance(sequence, str) or (sequence not in GENERIC and not sequence.startswith(expected_split+':')):
            raise ValueError('Real sequence IDs must be namespaced by their original source split')
        if not isinstance(row.get('reactant'), str) or not isinstance(row.get('target'), str):
            raise ValueError('Raw reactant and target strings are required')
        if not isinstance(row.get('rerank_eligible'), bool):
            raise ValueError('rerank_eligible must be a label-independent support boolean')
        samples = row.get('sample_count')
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError('sample_count must be a positive integer')
        previous, total, labels, raw_seen = math.inf, 0, 0, set()
        for rank, candidate in enumerate(row['candidates'], 1):
            n, raw = candidate['count'], candidate['smiles']
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n > previous:
                raise ValueError('Preserve descending positive frequency counts')
            if candidate['rank'] != rank or not isinstance(raw, str) or raw in raw_seen:
                raise ValueError('Preserve unique raw candidate strings and stable saved ordering')
            if not isinstance(candidate['correct'], bool) or candidate['correct'] != (raw == row['target']):
                raise ValueError('Correctness must use exact saved-writer target agreement')
            previous, total, labels = n, total+n, labels+candidate['correct']
            raw_seen.add(raw)
        if total > samples or labels > 1:
            raise ValueError('Inconsistent sample counts or correctness labels')
        if row.get('top1_correct') is not bool(row['candidates'] and row['candidates'][0]['correct']):
            raise ValueError('Baseline correctness differs from first saved candidate')
        keys = [('reactant', input_key(row['reactant']))]
        if sequence not in GENERIC:
            keys.append(('sequence', sequence))
        for key in keys:
            if key in connected and connected[key] != row['split_group']:
                raise ValueError('Connected rows have inconsistent split_group values')
            connected[key] = row['split_group']
    return rows


def load_records(path, expected_split):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return validate_records(rows, expected_split), hashlib.sha256(raw).hexdigest()


def read_manifest(path, expected_split):
    if path is None:
        return None
    manifest = json.loads(Path(path).read_text())
    required = {'row_id', 'source_row_id', 'source_split', 'split_group', 'sequence_id',
                'source_line_sha256', 'reactant_key'}
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('Audit manifest must be a nonempty metadata list')
    ids, sources = set(), set()
    for entry in manifest:
        if not required <= entry.keys() or entry['source_split'] != expected_split:
            raise ValueError('Audit manifest has missing metadata or incorrect source_split')
        if str(entry['row_id']) in ids or str(entry['source_row_id']) in sources:
            raise ValueError('Duplicate audit manifest identity')
        if not entry['split_group'] or (entry['sequence_id'] not in GENERIC and
                                        not entry['sequence_id'].startswith(expected_split+':')):
            raise ValueError('Audit manifest has invalid group/sequence metadata')
        ids.add(str(entry['row_id'])); sources.add(str(entry['source_row_id']))
    return manifest


def manifest_disjoint(sig, manifest):
    if manifest is None:
        return
    hashed = dict(sig, reactants=[hashlib.sha256(value.encode()).hexdigest() for value in sig['reactants']])
    other = {'groups': {r['split_group'] for r in manifest}, 'reactants': {r['reactant_key'] for r in manifest},
             'sequences': {r['sequence_id'] for r in manifest}-GENERIC}
    assert_disjoint(hashed, other, 'Development/audit manifest')


def manifest_membership(rows, manifest):
    if manifest is None:
        return
    expected = {str(r['row_id']): r for r in manifest}
    if len(rows) != len(manifest) or {str(r['row_id']) for r in rows} != set(expected):
        raise ValueError('Audit membership differs from frozen manifest')
    for row in rows:
        entry = expected[str(row['row_id'])]
        for key in ('source_row_id', 'source_split', 'split_group', 'sequence_id', 'source_line_sha256'):
            if row.get(key) != entry[key]:
                raise ValueError(f'Audit {key} differs from frozen manifest')
        if entry['reactant_key'] != hashlib.sha256(input_key(row['reactant']).encode()).hexdigest():
            raise ValueError('Audit reactant differs from frozen manifest')
        if 'selection_cohort' in entry and row.get('selection_cohort') != entry['selection_cohort']:
            raise ValueError('Audit selection_cohort differs from frozen manifest')


def features(row):
    """No access to target, correct, ranks, diagnostic labels, groups or provided features."""
    candidates = row['candidates'][:K]
    if not candidates:
        return None, {'reason': 'no_saved_candidates'}
    reactant = describe(row['reactant'])
    products = [describe(c['smiles']) for c in candidates]
    unsupported = [i+1 for i, (c, p) in enumerate(zip(candidates, products))
                   if not p['supported'] or c.get('parseable') is False]
    if not reactant['supported'] or unsupported or not row['rerank_eligible']:
        return None, {'reason': 'unsupported_input_or_candidate', 'unsupported_candidate_ranks': unsupported,
                      'input_supported': reactant['supported'], 'upstream_eligible': row['rerank_eligible'],
                      'candidate_valence_flags': [p.get('flags') for p in products]}
    rp, rc, rf = reactant['properties'], reactant['counts'], reactant['flags']
    matrix = []
    for candidate, product in zip(candidates, products):
        cp, cc, cf = product['properties'], product['counts'], product['flags']
        no_op = float(product['canonical'] == reactant['canonical'])
        values = [math.log(candidate['count']/row['sample_count']),
                  candidate['count']/sum(c['count'] for c in candidates),
                  DataStructs.TanimotoSimilarity(reactant['fp'], product['fp']), no_op]
        values += [cp[key]-rp[key] for key in ('abs_charge', 'charged_atoms', 'fragments', 'radicals', 'rings', 'aromatic_atoms')]
        values += [cp[key] for key in ('abs_charge', 'fragments', 'radicals')]
        values += [cc[key] for key in GROUPS]+[cc[key]-rc[key] for key in GROUPS]
        values += [cf[key] for key in FLAGS]+[cf[key]-rf[key] for key in FLAGS]
        values += list((1-no_op)*reactant['context'])+list(no_op*reactant['context'])
        same_graph = product['heavy_key'] == reactant['heavy_key']
        progress = [float(same_graph),
                    float(same_graph and reactant['context'][0] and cc['oxonium_h'] < rc['oxonium_h']),
                    float(same_graph and reactant['context'][0] and cc['highcoord_nhplus'] < rc['highcoord_nhplus'] and
                          cp['aromatic_atoms'] > rp['aromatic_atoms']),
                    float(same_graph and reactant['context'][1] and cc['hcl'] < rc['hcl'] and
                          cc['alkoxide_o']+cc['hydroxide_o'] < rc['alkoxide_o']+rc['hydroxide_o'] and
                          cc['chloride'] > rc['chloride']),
                    float(any(reactant['context'][2:4]) and cc['carbonyl_o'] < rc['carbonyl_o'] and
                          cc['tetrahedral_hetero_carbon'] > rc['tetrahedral_hetero_carbon']),
                    float(any(reactant['context'][4:]) and cp['rings'] > rp['rings'])]
        values += progress
        matrix.append(values)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(candidates), len(TARGETED_NAMES)) or not np.isfinite(matrix).all():
        return None, {'reason': 'unsupported_feature_values'}
    return matrix, {'reason': 'supported'}


def prepare(rows):
    pairs = [features(row) for row in rows]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def baseline(rows):
    return [0 if row['candidates'] else None for row in rows]


def family_matrix(matrix, family):
    return matrix[:, :len(GLOBAL_NAMES)] if family == 'learned_global' else matrix


def score_rule(matrix):
    lookup = {name: matrix[:, i] for i, name in enumerate(TARGETED_NAMES)}
    score = lookup['log_frequency'].copy()
    for flag in FLAGS[:4]:
        score += RULE_WEIGHTS['valence_flag']*lookup['candidate_'+flag]
    score += RULE_WEIGHTS['carbon_aluminum_charge_pair']*lookup['candidate_carbon_aluminum_charge_pair']
    for name in PROGRESS_NAMES[1:4]:
        score += RULE_WEIGHTS[name]*lookup[name]
    return score


def model_scores(matrices, model, family):
    if model is None:
        return [None]*len(matrices)
    nonempty = [family_matrix(m, family) for m in matrices if m is not None]
    if not nonempty:
        return [None]*len(matrices)
    values = model.decision_function(np.vstack(nonempty))
    result, offset = [], 0
    for matrix in matrices:
        if matrix is None:
            result.append(None)
        else:
            result.append(values[offset:offset+len(matrix)])
            offset += len(matrix)
    return result


def choose(rows, matrices, scores, config, method):
    result = baseline(rows)
    for i, (row, matrix, score) in enumerate(zip(rows, matrices, scores)):
        if matrix is None or score is None or config['alpha'] == 0:
            continue
        values = score if method == 'chemistry_rule' else (
            (1-config['alpha'])*matrix[:, 0]+config['alpha']*score)
        result[i] = stable_choice(values, config['margin'])
    return result


def pairwise_data(rows, matrices, family):
    x, y, weights, groups = [], [], [], set()
    paired_rows, skipped = 0, 0
    fit_targets = {}
    for row in rows:
        fit_targets.setdefault(input_key(row['reactant']), set()).add(row['target'])
    for row, matrix in zip(rows, matrices):
        if matrix is None:
            continue
        candidates = row['candidates'][:K]
        positive = next((i for i, c in enumerate(candidates) if c['correct']), None)
        negatives = [i for i, c in enumerate(candidates) if not c['correct']]
        if positive is None or not negatives:
            continue
        known = fit_targets[input_key(row['reactant'])]
        retained = [i for i in negatives if candidates[i]['smiles'] not in known]
        skipped += len(negatives)-len(retained)
        negatives = retained
        if not negatives:
            continue
        paired_rows += 1; groups.add(row['split_group'])
        matrix = family_matrix(matrix, family)
        for negative in negatives:
            delta = matrix[positive]-matrix[negative]
            x.extend((delta, -delta)); y.extend((1, 0)); weights.extend((.5/len(negatives),)*2)
    info = {'rows': len(rows), 'eligible_rows': sum(m is not None for m in matrices),
            'paired_rows': paired_rows, 'paired_groups': len(groups), 'candidate_pairs': len(x)//2,
            'weighting': 'one total weight per input row, equal over symmetric negative pairs',
            'negative_policy': 'both learned methods skip alternate exact targets observed for the same canonical input in FIT only',
            'contradictory_negative_pairs_skipped': skipped,
            'fit_inputs_with_multiple_targets': sum(len(v) > 1 for v in fit_targets.values())}
    return np.asarray(x), np.asarray(y), np.asarray(weights), info


def fit_models(rows, matrices, family):
    x, y, weights, info = pairwise_data(rows, matrices, family)
    if len(x) < 40 or info['paired_groups'] < 5:
        return {}, dict(info, status='insufficient_pairs_frequency_fallback')
    models, rejected = {}, []
    for c in CS:
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(
            C=c, fit_intercept=False, solver='liblinear', max_iter=2000, random_state=42))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            rejected.append(c)
        else:
            models[c] = model
    return models, dict(info, status='fitted' if models else 'convergence_failure_frequency_fallback', rejected_C=rejected)


def stratified_summaries(rows, outcomes):
    cohorts = sorted({str(row.get('selection_cohort', 'unspecified')) for row in rows})
    return {cohort: summarize([o for r, o in zip(rows, outcomes)
                              if str(r.get('selection_cohort', 'unspecified')) == cohort]) for cohort in cohorts}


def tune_method(rows, matrices, method, models):
    covered = [m is not None for m in matrices]
    selected = {'C': None, 'alpha': 0., 'margin': 0.}
    best_choices = baseline(rows)
    best = summarize(outcome_rows(rows, best_choices, K, covered))
    grid = [{'config': selected, 'summary': best}]
    if method == 'chemistry_rule':
        candidates = [(None, None, [score_rule(m) if m is not None else None for m in matrices])]
    else:
        candidates = [(c, model, model_scores(matrices, model, method)) for c, model in models.items()]
    chosen_model = None
    for c, model, scores in candidates:
        for alpha in ((1.,) if method == 'chemistry_rule' else ALPHAS):
            for margin in MARGINS:
                config = {'C': c, 'alpha': alpha, 'margin': margin}
                choices = choose(rows, matrices, scores, config, method)
                summary = summarize(outcome_rows(rows, choices, K, covered))
                grid.append({'config': config, 'summary': summary})
                if tuning_key(summary) > tuning_key(best):
                    selected, best, chosen_model, best_choices = config, summary, model, choices
    return {'config': selected, 'model': chosen_model, 'validation': best,
            'validation_strata': stratified_summaries(rows, outcome_rows(rows, best_choices, K, covered))}, grid


def weights_report(fitted, names):
    if fitted['model'] is None:
        return {'status': 'inactive_frequency_fallback', 'weights': []}
    scaler, logistic = (step[1] for step in fitted['model'].steps)
    standard = logistic.coef_[0]
    raw = standard/scaler.scale_
    order = np.argsort(-np.abs(standard), kind='stable')[:30]
    return {'status': 'active', 'weights': [{'feature': names[i], 'standardized_coefficient': float(standard[i]),
                                           'raw_coefficient': float(raw[i])} for i in order],
            'caution': 'Statistical descriptor associations; neither causal effects nor physical energies.'}


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ('targeted_reranker.py', 'pilot_reranker.py', 'rerank_analysis.py')}


def versions():
    return {'numpy': np.__version__, 'sklearn': sklearn.__version__,
            'rdkit': rdBase.rdkitVersion, 'joblib': joblib.__version__}


def fit_tune(args):
    implementation, runtime = implementation_hashes(), versions()
    audit_split = args.audit_split
    if audit_split not in ('val', 'test'):
        raise ValueError('Audit must be separate official val or test')
    if audit_split == 'test' and args.audit_manifest is None:
        raise ValueError('A precommitted metadata manifest is required for test assessment')
    fit, fit_hash = load_records(args.fit, 'train')
    tune, tune_hash = load_records(args.tune, 'val')
    fit_sig, tune_sig = signature(fit), signature(tune)
    assert_disjoint(fit_sig, tune_sig, 'Official train/tune')
    manifest_hash = file_hash(args.audit_manifest) if args.audit_manifest else None
    manifest = read_manifest(args.audit_manifest, audit_split)
    manifest_disjoint(fit_sig, manifest); manifest_disjoint(tune_sig, manifest)
    fit_matrices, fit_support = prepare(fit)
    tune_matrices, tune_support = prepare(tune)
    fitted, grids, training = {}, {}, {}
    fitted['chemistry_rule'], grids['chemistry_rule'] = tune_method(tune, tune_matrices, 'chemistry_rule', {})
    for method in ('learned_global', PRIMARY):
        models, training[method] = fit_models(fit, fit_matrices, method)
        fitted[method], grids[method] = tune_method(tune, tune_matrices, method, models)
    baseline_outcomes = outcome_rows(tune, baseline(tune), K, [m is not None for m in tune_matrices])
    validation = {'frequency': summarize(baseline_outcomes), **{m: f['validation'] for m, f in fitted.items()}}
    protocol = {'k': K, 'primary_method': PRIMARY, 'methods': list(METHODS),
                'C_grid': list(CS), 'alpha_grid': [0., *ALPHAS], 'margin_grid': list(MARGINS),
                'rule_weights_fixed_before_labels': RULE_WEIGHTS,
                'rule_tuning': 'only off/on and margin; fixed chemistry weights are never fitted',
                'learned_score': '(1-alpha)*log(count/sample_count)+alpha*pairwise_logistic_score',
                'learned_negative_policy': 'both models skip alternate exact targets for the same canonical input using FIT labels only',
                'selection': 'validation correct count, fewer changed rows, fewer harms, stable grid order',
                'primary_selection': 'learned_targeted predeclared, even when another method tunes better',
                'fallback': 'any unsupported top10 candidate/input, empty set, or validation alpha=0 retains baseline',
                'inference_gating': 'input/candidate support only; never target presence, labels, or reference graph',
                'features': {'learned_global': GLOBAL_NAMES, PRIMARY: TARGETED_NAMES},
                'feature_interpretation': 'Unmapped molecule descriptors and contrasts; no asserted atom correspondence.',
                'global_similarity': 'Morgan radius 2, 512 bits, Tanimoto similarity only',
                'protonation_groups': 'Structural environments, not measured pKa or a universal basicity order.',
                'no_op': 'canonical structural equality for features; exact saved strings retained for strict labels',
                'audit_split': audit_split, 'bootstrap_resamples': 10000, 'bootstrap_seed': 20261028,
                'denominator': 'all rows, including baseline-correct, absent-target, unsupported and empty sets',
                'benchmark_resume_automatic': False}
    inputs = {'fit_records_sha256': fit_hash, 'tune_records_sha256': tune_hash,
              'audit_manifest_sha256': manifest_hash}
    frozen = {'version': VERSION, 'protocol': protocol, 'inputs': inputs,
              'signatures': {'fit': fit_sig, 'tune': tune_sig}, 'audit_manifest': manifest,
              'development_rows': {role: [{key: row.get(key) for key in
                                          ('row_id', 'source_row_id', 'source_split', 'split_group',
                                           'sequence_id', 'selection_cohort')} for row in records]
                                   for role, records in (('fit', fit), ('tune', tune))},
              'implementation_sha256': implementation, 'runtime_versions': runtime,
              'models': fitted, 'frozen_at': time.time()}
    if file_hash(args.fit) != fit_hash or file_hash(args.tune) != tune_hash:
        raise ValueError('Development records changed during fitting')
    if args.audit_manifest and file_hash(args.audit_manifest) != manifest_hash:
        raise ValueError('Audit manifest changed during fitting')
    if implementation_hashes() != implementation or versions() != runtime:
        raise ValueError('Implementation/runtime changed during fitting')
    out = fresh_directory(args.out)
    joblib.dump(frozen, out/'frozen.joblib')
    receipt = {key: value for key, value in frozen.items() if key != 'models'}
    receipt.update(model_sha256=file_hash(out/'frozen.joblib'), training=training, validation=validation,
                   validation_strata={'frequency': stratified_summaries(tune, baseline_outcomes),
                                      **{m: f['validation_strata'] for m, f in fitted.items()}},
                   selected_configurations={m: f['config'] for m, f in fitted.items()},
                   support={'fit': dict(Counter(s['reason'] for s in fit_support)),
                            'tune': dict(Counter(s['reason'] for s in tune_support))},
                   top_weights={m: weights_report(fitted[m], protocol['features'][m])
                                for m in ('learned_global', PRIMARY)},
                   status='frozen_before_separate_audit')
    write_json(out/'frozen.json', receipt)
    write_json(out/'validation-grid.json', grids)
    for name in ('frozen.joblib', 'frozen.json', 'validation-grid.json'):
        (out/name).chmod(0o444)
    print(json.dumps({'model': str(out/'frozen.joblib'), 'primary_method': PRIMARY,
                      'validation': validation, 'training': training}, indent=2), flush=True)
    return receipt


def evidence_status(summary):
    interval = summary['paired_group_bootstrap']
    discordant = summary['corrected']+summary['harmed']
    status = ('negative' if summary['net_accuracy_delta'] < 0 else
              'tiny' if interval['group_count'] < 30 or discordant < 20 else
              'positive_pilot_signal' if summary['net_accuracy_delta'] >= .01 and interval['lower'] > 0 else
              'inconclusive')
    return {'status': status, 'discordant_rows': discordant, 'automatic_benchmark_resumption': False,
            'criteria': 'positive requires >=1 pp net, CI lower>0, >=30 groups and >=20 discordant rows'}


def evaluate(args):
    model_path = Path(args.model)
    receipt = json.loads((model_path.parent/'frozen.json').read_text())
    model_hash = file_hash(model_path)
    if model_hash != receipt['model_sha256']:
        raise ValueError('Frozen model changed')
    frozen = joblib.load(model_path)
    if frozen['version'] != VERSION or frozen['protocol']['features'] != {
            'learned_global': GLOBAL_NAMES, PRIMARY: TARGETED_NAMES}:
        raise ValueError('Frozen version/feature schema mismatch')
    if frozen['implementation_sha256'] != implementation_hashes() or frozen['runtime_versions'] != versions():
        raise ValueError('Frozen implementation/runtime changed')
    for key, value in frozen.items():
        if key != 'models' and json.dumps(value, sort_keys=True) != json.dumps(receipt.get(key), sort_keys=True):
            raise ValueError('Frozen receipt disagrees with model metadata')
    for method, fitted in frozen['models'].items():
        if fitted['config'] != receipt['selected_configurations'][method]:
            raise ValueError('Frozen selected configuration changed')
    rows, records_hash = load_records(args.records, frozen['protocol']['audit_split'])
    manifest_membership(rows, frozen['audit_manifest'])
    sig = signature(rows)
    for role in ('fit', 'tune'):
        assert_disjoint(frozen['signatures'][role], sig, role+'/audit')
    binding = {'model_sha256': model_hash, 'records_sha256': records_hash,
               'audit_manifest_sha256': frozen['inputs']['audit_manifest_sha256'],
               'output': str(Path(args.out).resolve())}
    claim = model_path.parent/'evaluation-claim.json'
    if claim.exists():
        if json.loads(claim.read_text()) != binding:
            raise ValueError('Frozen model already bound to another evaluation; mutation forbidden')
        artifacts_path = Path(args.out)/'artifact-receipt.json'
        if not artifacts_path.exists():
            raise ValueError('Prior evaluation incomplete; inspect it without changing the audit cohort')
        artifacts = json.loads(artifacts_path.read_text())
        if set(artifacts) != {'comparison.json', 'decisions.jsonl'} or any(
                not (Path(args.out)/name).exists() or file_hash(Path(args.out)/name) != digest
                for name, digest in artifacts.items()):
            raise ValueError('Saved evaluation artifacts changed')
        report = json.loads((Path(args.out)/'comparison.json').read_text())
        if report['binding'] != binding:
            raise ValueError('Saved evaluation binding changed')
        print(json.dumps(report, indent=2), flush=True)
        return report
    out = fresh_directory(args.out)
    with claim.open('x') as stream:
        json.dump(binding, stream, sort_keys=True)
    matrices, support = prepare(rows)
    covered = [m is not None for m in matrices]
    selected = {'frequency': baseline(rows)}
    for method, fitted in frozen['models'].items():
        scores = ([score_rule(m) if m is not None else None for m in matrices] if method == 'chemistry_rule'
                  else model_scores(matrices, fitted['model'], method))
        selected[method] = choose(rows, matrices, scores, fitted['config'], method)
    detailed = {method: outcome_rows(rows, selected[method], K, covered) for method in METHODS}
    methods = {}
    for method, outcomes in detailed.items():
        for outcome, status in zip(outcomes, support):
            outcome['decision_reason'] = (
                'frequency_baseline' if method == 'frequency' else
                status['reason'] if status['reason'] != 'supported' else
                'validation_frequency_fallback' if frozen['models'][method]['config']['alpha'] == 0 else
                'switched' if outcome['changed'] else 'no_margin_supported_switch')
        reasons = Counter(o['decision_reason'] for o in outcomes)
        summary = summarize(outcomes)
        summary.update(net_accuracy_change_pp=100*summary['net_accuracy_delta'],
                       eligible_coverage=summary['covered_rows']/len(rows),
                       no_saved_candidates=sum(not r['candidates'] for r in rows),
                       unsupported_fallback_rows=sum(s['reason'] == 'unsupported_input_or_candidate' for s in support),
                       baseline_retained_rows=sum(not o['changed'] for o in outcomes),
                       decision_reasons=dict(reasons),
                       active_scoring_rows=(sum(covered) if method != 'frequency' and
                                            frozen['models'][method]['config']['alpha'] > 0 else 0),
                       paired_group_bootstrap=cluster_bootstrap(outcomes),
                       selection_strata=stratified_summaries(rows, outcomes))
        methods[method] = summary
    ablation_pairs = [{'split_group': row['split_group'],
                       'baseline_correct': global_outcome['correct'], 'correct': targeted_outcome['correct']}
                      for row, global_outcome, targeted_outcome in
                      zip(rows, detailed['learned_global'], detailed[PRIMARY])]
    over_global = sum(p['correct'] and not p['baseline_correct'] for p in ablation_pairs)
    below_global = sum(p['baseline_correct'] and not p['correct'] for p in ablation_pairs)
    with (out/'decisions.jsonl').open('w') as stream:
        for i, row in enumerate(rows):
            record = {key: row.get(key) for key in ('row_id', 'source_row_id', 'source_split', 'split_group', 'selection_cohort')}
            record.update(support=support[i], methods={m: detailed[m][i] for m in METHODS})
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
    report = {'version': VERSION, 'binding': binding, 'primary_method': PRIMARY,
              'status': 'group_disjoint_official_validation_audit' if frozen['protocol']['audit_split'] == 'val'
                        else 'frozen_test_subset_assessment',
              'rows': len(rows), 'groups': len(sig['groups']), 'k': K, 'methods': methods,
              'targeted_minus_global_ablation': {
                  'rows': len(rows), 'improved': over_global, 'harmed': below_global,
                  'net_correct': over_global-below_global,
                  'net_accuracy_change_pp': 100*(over_global-below_global)/len(rows),
                  'paired_group_bootstrap': cluster_bootstrap(ablation_pairs),
                  'interpretation': 'Prespecified secondary comparison, not used to select the audit primary.'},
              'primary_evidence': evidence_status(methods[PRIMARY]), 'evaluated_at': time.time(),
              'top_weights': receipt['top_weights'],
              'notes': ['The primary targeted method is fixed before audit, regardless of ablation performance.',
                        'All four methods use the same saved top10 raw strings and retain every row.',
                        'Unsupported chemistry keeps frequency for every nonbaseline method.',
                        'Target absence is used only for fitting availability and reporting, never inference gating.',
                        'Tune selection strata and separate audit strata are reported without outcome-based cohort selection.',
                        'Unchanged_wrong is correctness status; wrong-to-wrong candidate switches are also counted.',
                        'Group-bootstrap intervals are descriptive, not adjusted for multiple comparisons.',
                        'Features are qualitative structural hypotheses, not energies, pKa measurements or atom mappings.',
                        'The caller must preserve outcome-blind cohort selection; this code verifies metadata and group disjointness.']}
    if file_hash(args.records) != records_hash or file_hash(model_path) != model_hash:
        raise ValueError('Frozen inputs changed during evaluation')
    write_json(out/'comparison.json', report)
    write_json(out/'artifact-receipt.json', {name: file_hash(out/name) for name in ('comparison.json', 'decisions.jsonl')})
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit-tune')
    for name in ('fit', 'tune', 'out'):
        fit.add_argument('--'+name, type=Path, required=True)
    fit.add_argument('--audit-manifest', type=Path)
    fit.add_argument('--audit-split', choices=('val', 'test'), default='val')
    fit.set_defaults(func=fit_tune)
    evaluation = sub.add_parser('evaluate')
    for name in ('records', 'model', 'out'):
        evaluation.add_argument('--'+name, type=Path, required=True)
    evaluation.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
