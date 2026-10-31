#!/usr/bin/env python3
"""Audit labels in an inspected prefix; no inference or unprocessed test labels."""
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from diagnose_rank_failures import chemistry, reference
from rerank_analysis import PARSER
RDLogger.DisableLog('rdApp.*')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_functions(path):
    """Load the actual project input helpers, without requiring local torch."""
    nodes = []
    functions = {'bond_features', 'count_lone_pairs', 'get_BE_matrix', 'smi2vocabid'}
    variables = {'ELEM_LIST', 'bt_to_electron', 'tbl', 'ps', 'atom2idx_dict'}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in variables or
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'ps'
            for t in node.targets):
            nodes.append(node)
    ns = {'Chem': Chem, 'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['smi2vocabid'], ns['get_BE_matrix']


def normalized(mol):
    return reference(Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True))


@lru_cache(maxsize=50000)
def fragments(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None:
        return None
    fs = Chem.GetMolFrags(mol, asMols=True)
    organic = [(f.GetNumHeavyAtoms(), normalized(f)) for f in fs
               if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    winners = [s for n, s in organic if n == max(x[0] for x in organic)] if organic else []
    return {'all': dict(Counter(normalized(f) for f in fs)),
            'largest': winners[0] if len(winners) == 1 else None,
            'formula': rdMolDescriptors.CalcMolFormula(mol)}


@lru_cache(maxsize=20000)
def hydrogen_connectivity(smiles):
    """Preserve attached H counts as node colors, ignoring bond order/charge."""
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles, PARSER))
    for atom in mol.GetAtoms():
        h = atom.GetTotalNumHs(includeNeighbors=True)
        atom.SetAtomMapNum(0); atom.SetIsotope(100+h); atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0); atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True); atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE); bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


class ResonanceDeadline(Chem.ResonanceMolSupplierCallback):
    def __init__(self, seconds):
        super().__init__(); self.deadline = time.monotonic()+seconds

    def __call__(self):
        return time.monotonic() < self.deadline


@lru_cache(maxsize=10000)
def resonance_forms(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol.GetNumHeavyAtoms() > 128:
        return frozenset(), 'heavy_atom_limit'
    try:
        supplier = Chem.ResonanceMolSupplier(mol, Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION, 129)
        callback = ResonanceDeadline(.25); supplier.SetProgressCallback(callback)
        result = {reference(smiles)}
        for i, form in enumerate(supplier):
            if i >= 128:
                return frozenset(result), 'structure_limit'
            result.add(normalized(form))
        return frozenset(result), 'canceled' if supplier.WasCanceled() else 'complete'
    except Exception as error:
        return frozenset(), 'unsupported_'+type(error).__name__


@lru_cache(maxsize=15000)
def compare_states(left, right):
    lc, rc = chemistry(left), chemistry(right)
    if not lc['parseable'] or not rc['parseable']:
        return {'category': 'unsupported_saved_string'}
    same = reference(left) == reference(right)
    bonds = lc.get('heavy_bond_signature') is not None and lc.get('heavy_bond_signature') == rc.get('heavy_bond_signature')
    conn = lc.get('connectivity_signature') is not None and lc.get('connectivity_signature') == rc.get('connectivity_signature')
    lf, rf = fragments(left), fragments(right)
    same_h = hydrogen_connectivity(left) == hydrogen_connectivity(right) if conn else False
    resonant, status = False, 'different_H_attachment_or_connectivity'
    if not same and same_h and lf['formula'] == rf['formula']:
        forms, status = resonance_forms(left); resonant = reference(right) in forms
        if not resonant:
            forms, other = resonance_forms(right)
            resonant = reference(left) in forms; status += '/'+other
    category = ('serialization_only' if same else 'rdkit_resonance_related' if resonant else
                'same_heavy_bonds_H_charge_difference' if bonds else
                'same_connectivity_bond_order_or_H_charge_difference' if conn else
                'different_heavy_connectivity')
    return {'category': category, 'same_heavy_bonds': bonds, 'same_heavy_connectivity': conn,
            'same_H_attachment_connectivity': same_h, 'resonance_status': status,
            'largest_left': lf['largest'], 'largest_right': rf['largest'],
            'formula_equal': lf['formula'] == rf['formula']}


def mapped_change(left, right):
    """Only used for witness pairs with exactly identical model input arrays."""
    def indexed(s):
        mol = Chem.MolFromSmiles(s, PARSER)
        atoms = {a.GetAtomMapNum(): a for a in mol.GetAtoms()}
        bonds = {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))):
                 b.GetBondTypeAsDouble() for b in mol.GetBonds()}
        hs = {i: sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors())+a.GetTotalNumHs()
              for i, a in atoms.items() if a.GetAtomicNum() != 1}
        return atoms, bonds, hs
    la, lb, lh = indexed(left); ra, rb, rh = indexed(right)
    if set(la) != set(ra) or any(la[i].GetAtomicNum() != ra[i].GetAtomicNum() for i in la):
        return {'status': 'incompatible_maps'}
    heavy, hydrogen = [], []
    for pair in sorted(set(lb) | set(rb)):
        if lb.get(pair, 0) == rb.get(pair, 0): continue
        dest = hydrogen if any(la[i].GetAtomicNum() == 1 for i in pair) else heavy
        dest.append({'atoms': list(pair), 'elements': [la[i].GetSymbol() for i in pair],
                     'left_order': lb.get(pair, 0), 'right_order': rb.get(pair, 0)})
    return {'status': 'same_input_indices', 'heavy_bond_changes': heavy,
            'hydrogen_bond_changes': hydrogen, 'changed_heavy_H_counts': [i for i in lh if lh[i] != rh[i]],
            'charge_changes': [{'atom': i, 'element': la[i].GetSymbol(), 'left': la[i].GetFormalCharge(),
                                'right': ra[i].GetFormalCharge()} for i in la
                               if la[i].GetFormalCharge() != ra[i].GetFormalCharge()]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--data-utils', type=Path, default=Path('utils/data_utils.py'))
    p.add_argument('--input-index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Preserve existing audit; use a new output path')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.records.open()]
    assert len(rows) == 20152 and all(r['source_row_id'] == i for i, r in enumerate(rows))
    tokenize, matrix = input_functions(args.data_utils)
    cache, exact, targets, witnesses = {}, defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    graphs, terminals = defaultdict(lambda: defaultdict(set)), defaultdict(set)
    for index, r in enumerate(rows):
        raw = r['reaction'].split('>>')[0]
        if raw not in cache:
            tokens, n = tokenize(raw); be = matrix(raw); assert be.shape == (n, n)
            cache[raw] = hashlib.sha256(tokens.tobytes()+be.tobytes()).hexdigest()
        r['model_input_hash'] = cache[raw]
        exact[cache[raw]][r['target']] += 1; targets[r['reactant']][r['target']] += 1
        witnesses[(r['reactant'], r['target'])].append(r)
        graphs[r['sequence_id']][r['reactant']].add(r['target'])
        if r['reactant'] == r['target']: terminals[r['sequence_id']].add(r['target'])
        if (index+1) % 5000 == 0: print('Checked model inputs', index+1, flush=True)

    @lru_cache(maxsize=15000)
    def reachable(sequence, state):
        visited, todo = set(), [state]
        while todo:
            node = todo.pop()
            if node in visited: continue
            visited.add(node); todo.extend(graphs[sequence].get(node, set())-visited)
        return frozenset(visited), frozenset(visited & terminals[sequence])

    totals, group_rows, errors, ambiguous, residual = Counter(), defaultdict(list), [], [], []
    graph_counts = Counter()
    with (args.out/'row-audit.jsonl').open('w') as stream:
        for r in rows:
            first = r['candidates'][0] if r['candidates'] else None
            top = first['smiles'] if first else None
            known, ek = targets[r['reactant']], exact[r['model_input_hash']]
            pf, tf = fragments(top) if top else None, fragments(r['target'])
            available = bool(pf and pf['largest'] and tf['largest'])
            flags = {'strict_top1': r['top1_correct'], 'strict_top10': any(c['correct'] for c in r['candidates']),
                'observed_top1': top in known, 'observed_top10': any(c['smiles'] in known for c in r['candidates']),
                'exact_input_observed_top1': top in ek, 'exact_input_observed_top10': any(c['smiles'] in ek for c in r['candidates']),
                'largest_organic_available': available, 'largest_organic_equal': bool(available and pf['largest'] == tf['largest']),
                'multiple_observed_targets': len(known) > 1, 'no_saved_candidate': first is None,
                'unsupported_candidate_row': not r['rerank_eligible']}
            totals.update(k for k, v in flags.items() if v)
            group_rows['multiple_targets' if len(known)>1 else 'one_observed_target'].append(flags)
            item = dict(row_id=r['row_id'], test_line=r['row_id']+1, sequence_id=r['sequence_id'],
                        model_input_hash=r['model_input_hash'], target_rank=r['target_rank'], **flags)
            stream.write(json.dumps(item)+'\n')
            if r['top1_correct']: continue
            peers = sorted(witnesses.get((r['reactant'], top), []), key=lambda w: (
                w['model_input_hash'] != r['model_input_hash'], w['sequence_id'] != r['sequence_id'],
                abs(w['row_id']-r['row_id']), w['row_id']))
            peer = peers[0] if peers else None
            item.update(input=r['reactant'], reference=r['target'], top1=top, candidates=r['candidates'],
                top1_count=first['count'] if first else 0, comparison=compare_states(top, r['target']) if top else {'category':'no_saved_candidate'},
                observed_reference_targets=dict(known), witness_lines=[w['row_id']+1 for w in peers],
                same_sequence_alternative=any(w['sequence_id']==r['sequence_id'] for w in peers),
                top1_no_change=r['top1_no_op'], reference_no_change=r['target_no_op'],
                selected_witness_line=peer['row_id']+1 if peer else None)
            if peer and peer['model_input_hash'] == r['model_input_hash']:
                item['mapped_difference'] = mapped_change(r['reaction'].split('>>')[1], peer['reaction'].split('>>')[1])
            if item['same_sequence_alternative']:
                rn, rt = reachable(r['sequence_id'], r['target']); pn, pt = reachable(r['sequence_id'], top)
                rl = {fragments(s)['largest'] for s in rt}-{None}; pl = {fragments(s)['largest'] for s in pt}-{None}
                g = {'both_have_recorded_terminal': bool(rt and pt), 'common_full_terminal': bool(rt & pt),
                     'common_largest_terminal_fragment': bool(rl & pl), 'top1_downstream_of_reference': top in rn,
                     'reference_downstream_of_top1': r['target'] in pn}
                item['recorded_graph'] = g; graph_counts.update(k for k, v in g.items() if v)
                graph_counts['same_sequence_cases'] += 1
            errors.append(item); (ambiguous if flags['observed_top1'] else residual).append(item)
            if len(errors) % 500 == 0: print('Characterized mismatches', len(errors), flush=True)
    for name, cohort in [('all-top1-errors', errors), ('competing-recorded-targets', ambiguous), ('unmatched-top1', residual)]:
        with (args.out/f'{name}.jsonl').open('w') as stream:
            for r in cohort: stream.write(json.dumps(r)+'\n')
    metadata = {'processed': Counter(), 'unprocessed': Counter()}
    for line in args.input_index.open():
        r = json.loads(line); tag = str(r['raw_sequence_id']); tag = 'numeric_sequence' if tag.isdigit() else tag
        metadata['processed' if r['source_row_id'] < len(rows) else 'unprocessed'][tag] += 1
    summary = {'status': 'retrospective_label_audit_not_new_benchmark', 'rows': len(rows),
        'counts': dict(totals), 'strict_top1_errors': len(errors), 'competing_errors': len(ambiguous), 'unmatched_errors': len(residual),
        'metrics': {k: {'matches': totals[k], 'rows':len(rows), 'fraction':totals[k]/len(rows)} for k in
                    ['strict_top1','strict_top10','observed_top1','observed_top10','exact_input_observed_top1','exact_input_observed_top10']},
        'largest_fragment_diagnostic': {'matches':totals['largest_organic_equal'],'available':totals['largest_organic_available']},
        'recorded_graph_comparisons': dict(graph_counts),
        'error_subsets': {name: {'rows':len(cohort), 'target_in_top10':sum(r['strict_top10'] for r in cohort),
            'exact_input_alternative':sum(r['exact_input_observed_top1'] for r in cohort),
            'same_sequence_alternative':sum(r['same_sequence_alternative'] for r in cohort),
            'largest_organic_equal':sum(r['largest_organic_equal'] for r in cohort),
            'top1_no_change':sum(r['top1_no_change'] for r in cohort), 'reference_no_change':sum(r['reference_no_change'] for r in cohort),
            'categories':dict(Counter(r['comparison']['category'] for r in cohort))}
            for name,cohort in [('all',errors),('competing',ambiguous),('unmatched',residual)]},
        'input_target_structure': {'canonical_inputs':len(targets),'exact_model_inputs':len(exact),
            'multiple_target_canonical_inputs':sum(len(c)>1 for c in targets.values()),
            'multiple_target_exact_inputs':sum(len(c)>1 for c in exact.values()),
            'distinct_input_target_pairs':sum(len(c) for c in targets.values()),
            'best_fixed_choice_canonical':sum(max(c.values()) for c in targets.values())/len(rows),
            'best_fixed_choice_exact':sum(max(c.values()) for c in exact.values())/len(rows)},
        'input_cohorts': {name: {'rows':len(values),**{k:sum(r[k] for r in values) for k in totals}} for name,values in group_rows.items()},
        'network_structure': {'sequence_ids':len(graphs),'ids_with_branching':sum(any(len(v-{k})>1 for k,v in g.items()) for g in graphs.values()),
            'ids_with_multiple_terminals':sum(len(t)>1 for t in terminals.values()),'self_loop_rows':sum(r['target_no_op'] for r in rows)},
        'metadata_only_source_distribution': metadata,
        'sources': {'records_sha256':sha(args.records),'data_utils_sha256':sha(args.data_utils),
                    'input_index_sha256':sha(args.input_index),'script_sha256':sha(__file__),'rdkit':rdBase.rdkitVersion},
        'limitations': ['No unprocessed test target labels were read.',
            'Observed-target agreement is a retrospective lookup diagnostic, not new benchmark accuracy.',
            'Recorded alternatives are not independent experimental verification of chemical feasibility.',
            'Largest organic fragment agreement is a proxy, not identification of the desired final product.',
            'Reference-graph reachability is not a model rollout.',
            'Resonance positives are RDKit representation matches; bounded nonmatches do not prove non-equivalence.',
            'Label frequencies count annotations, not experimental yields or branch probabilities.']}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
