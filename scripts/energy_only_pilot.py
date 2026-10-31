#!/usr/bin/env python3
"""Prepare or summarize a held, validation-only energy-ranking diagnostic.

Preparation performs graph checks, never conformer generation or xTB calls.
Only the seven explicitly authorized validation sources are opened as data.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D
from audit_recorded_targets import input_functions
from phase0_reranking_audit import model_input_hash
from rerank_analysis import PARSER, canonical
import score_xtb_candidates as xtb

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/reranking/full-validation-test20k-20261029'
VAL = ROOT / 'data/flower_new_dataset/val.txt'
SOURCES = [BASE / 'cache/val.jsonl'] + [BASE / f'inference/records/val{i:03}.jsonl' for i in range(5)]
STRATA = ('another_recorded_outcome', 'no_recorded_outcome_match')
SEED = 20261031
HARTREE_KCAL = 627.5094740631
SOLVENT_SMILES = {'thf': 'C1CCOC1', 'dioxane': 'C1COCCO1', 'dmf': 'CN(C)C=O',
    'dmso': 'CS(C)=O', 'acetonitrile': 'CC#N', 'ch2cl2': 'ClCCl', 'chcl3': 'ClC(Cl)Cl',
    'toluene': 'Cc1ccccc1', 'benzene': 'c1ccccc1', 'ether': 'CCOCC',
    'ethylacetate': 'CCOC(C)=O', 'acetone': 'CC(C)=O', 'methanol': 'CO',
    'ethanol': 'CCO', 'water': 'O', 'hexane': 'CCCCCC'}


def sha(path):
    return xtb.file_hash(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    xtb.atomic_json(path, value)


def jsonl(path, values):
    xtb.write_jsonl_atomic(path, values)


def protocol(solvent_rule):
    return {'version': 'flower-energy-only-failure-recovery-v1', 'seed': SEED,
        'question': 'Does energy-only ranking recover the recorded reference from saved top-10 failure pools?',
        'execution_holds': {k: True for k in ('physical_calculations', 'reranker_training',
            'new_inference', 'cloud_rentals', 'reserved_test_outcomes')},
        'observations': 'official validation only; no test-prefix observations',
        'catalog': 'all 21723 official-validation targets, grouped by evaluator-normalized complete input',
        'exact_input_catalog': 'also reported; never conflated with normalized-input catalog',
        'selection': {'quota_per_stratum': 25, 'one_row_per_connected_group': True,
            'eligibility': 'reference ranks 2-10; graph-verified single proton relocation/no-change choices; complete static physical support; declared solvent rule',
            'randomization': 'seeded group-uniform shuffle within strata, then seeded row shuffle; round-robin strata with globally unique groups',
            'replacement_after_energies': False, 'rarity_used': False},
        'ranking': 'ascending sum of fragment minimum prescribed-conformer ALPB total energies; no frequency term or fitted parameters',
        'tie_break': 'lexicographic immutable candidate SMILES, independent of frequency and labels',
        'near_tie_kcal_mol': 1.0,
        'near_tie_policy': 'report only; no tuned gap cutoff; exact numerical ties use shared minimum rank and a deterministic operational choice',
        'physical': {'method': 'GFN2-xTB', 'solvation': 'ALPB', 'solvent_rule': solvent_rule,
            'reference_state': 'bar1mol', 'electronic_temperature_K': 300.,
            'reaction_temperature': 'unavailable; electronic temperature is not the reaction temperature',
            'conformers': 2, 'embedding': 'ETKDGv3', 'embedding_seed': 42,
            'optimization': 'normal', 'require_all_conformers': True,
            'spin': 'closed-shell supported main-group fragments only; otherwise unsupported',
            'max_heavy_atoms_per_fragment': 40, 'strict_elements': sorted(xtb.SUPPORTED_ELEMENTS),
            'isotope_policy': 'unsupported by current XYZ pipeline',
            'score': 'ALPB total energy includes solvation once; isolated fragment sums, not full Gibbs energies',
            'fallback': 'any unreliable candidate makes whole row unresolved; retain frequency prediction'},
        'budget': {'workers': 2, 'threads_per_worker': 1, 'elapsed_seconds': 7200,
            'fragment_wall_seconds_including_two_conformers': 20, 'max_unique_jobs': 720,
            'new_rentals': False, 'automatic_retries': False},
        'assessment': {'denominators': ['all selected rows', 'complete-pool rows'],
            'overall_accuracy_claim': False, 'chemical_assessment': 'separate, not established by reference recovery'},
        'conversion_hartree_to_kcal_mol': HARTREE_KCAL}


@lru_cache(maxsize=15000)
def fragments(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, PARSER))
    return tuple(sorted(canonical(Chem.MolToSmiles(f)) for f in Chem.GetMolFrags(mol, asMols=True)))


def _heavy_graph(parts):
    mol = Chem.AddHs(Chem.MolFromSmiles('.'.join(parts), PARSER))
    graph = Chem.RWMol(); indices = {}; annotations = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            if a.GetDegree() != 1 or a.GetNeighbors()[0].GetAtomicNum() == 1:
                raise ValueError('Changed free proton/hydride/H2 is outside this attachment test')
            continue
        new = Chem.Atom(a.GetAtomicNum()); new.SetNoImplicit(True)
        new.SetIsotope(a.GetIsotope()); new.SetIsAromatic(a.GetIsAromatic())
        indices[a.GetIdx()] = graph.AddAtom(new)
        annotations.append({'element': a.GetSymbol(), 'atomic_number': a.GetAtomicNum(),
            'hydrogens': sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()),
            'charge': a.GetFormalCharge(), 'radicals': a.GetNumRadicalElectrons()})
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in indices and j in indices:
            graph.AddBond(indices[i], indices[j], b.GetBondType())
    return graph.GetMol(), annotations


@lru_cache(maxsize=10000)
def proton_relation(left, right):
    """Conservative graph proof across all enumerated heavy-graph bijections.

    Uses only these two unmapped states, never the reference's atom maps.
    This verifies proton relocation in graphs, not chemical/kinetic preference.
    """
    if canonical(left) == canonical(right):
        return {'status': 'no_change'}
    if xtb.inventory(left) != xtb.inventory(right):
        return {'status': 'inventory_or_net_charge_change'}
    a, b = Counter(fragments(left)), Counter(fragments(right))
    common = a & b; a -= common; b -= common
    try:
        ga, aa = _heavy_graph(list(a.elements())); gb, bb = _heavy_graph(list(b.elements()))
    except Exception:
        return {'status': 'unsupported_changed_fragment'}
    if ga.GetNumAtoms() != gb.GetNumAtoms() or ga.GetNumBonds() != gb.GetNumBonds():
        return {'status': 'heavy_graph_changed'}
    matches = gb.GetSubstructMatches(ga, uniquify=False, maxMatches=257, useChirality=False)
    if not matches:
        return {'status': 'heavy_graph_changed'}
    if len(matches) >= 257:
        return {'status': 'mapping_enumeration_limit'}
    edits = []
    for mapping in matches:
        changed = []
        for i, j in enumerate(mapping):
            x, y = aa[i], bb[j]
            dh, dq = y['hydrogens'] - x['hydrogens'], y['charge'] - x['charge']
            if x['radicals'] or y['radicals']:
                return {'status': 'radical_electronic_change'}
            if dh or dq:
                changed.append({'source_heavy_index': i, 'candidate_heavy_index': j,
                    'element': x['element'], 'delta_H': dh, 'delta_charge': dq})
        if len(changed) != 2 or sorted(e['delta_H'] for e in changed) != [-1, 1] or any(e['delta_H'] != e['delta_charge'] for e in changed):
            return {'status': 'not_invariant_single_proton_relocation', 'mappings_checked': len(matches)}
        donor = next(e for e in changed if e['delta_H'] == -1)
        acceptor = next(e for e in changed if e['delta_H'] == 1)
        if donor['element'] not in {'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'} or acceptor['element'] not in {'N', 'O', 'S'}:
            return {'status': 'outside_heteroatom_proton_choice_scope'}
        edits.append(changed)
    return {'status': 'verified_graph_proton_relocation', 'mappings_checked': len(matches),
        'example_mapping_edits': edits[0], 'changed_source_fragments': list(a.elements()),
        'changed_candidate_fragments': list(b.elements()),
        'qualification': 'graph attachment and charge proof, not an independent chemical preference judgment'}


def chemical_eligibility(row):
    first = row['candidates'][0]['smiles']
    reference = proton_relation(row['reactant'], row['target'])
    winner = proton_relation(row['reactant'], first)
    allowed = {'no_change', 'verified_graph_proton_relocation'}
    eligible = (reference['status'] in allowed and winner['status'] in allowed and reference['status'] != winner['status']) or (
        reference['status'] == winner['status'] == 'verified_graph_proton_relocation')
    return bool(eligible), {'input_to_reference': reference, 'input_to_frequency_winner': winner}


def solvent_context(row, rule):
    # This is an explicitly declared proxy when authorized, never measured solvent metadata.
    lookup = {canonical(Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(s)))): name
              for name, s in SOLVENT_SMILES.items()}
    parts = Counter(fragments(row['reactant']))
    present = sorted({lookup[p] for p in parts if p in lookup})
    evidence = {'recognized_input_solvents': present, 'explicit_bulk_solvent_metadata': False,
                'rule': rule, 'solvent': None, 'status': 'unresolved_solvent_context'}
    if rule == 'documented':
        return evidence
    if len(present) != 1 or present[0] == 'water':
        return evidence
    solvent = present[0]; smi = next(p for p, name in lookup.items() if name == solvent)
    if any(Counter(fragments(c['smiles']))[smi] != parts[smi] for c in row['candidates']):
        evidence['reason'] = 'recognized_solvent_changes_in_a_saved_candidate'
        return evidence
    evidence.update(status='declared_nominal_assumption', solvent=solvent,
        evidence='sole recognized organic solvent, unchanged with equal multiplicity in the input and every retained candidate; bulk identity/concentration unmeasured')
    return evidence


def static_pool_check(row):
    args = SimpleNamespace(max_heavy_atoms=40, strict_elements=True)
    failures = []
    for c in row['candidates']:
        if xtb.inventory(c['smiles']) != xtb.inventory(row['reactant']):
            failures.append({'rank': c['rank'], 'reason': 'inventory_or_net_charge_mismatch'})
        for part in fragments(c['smiles']):
            reason = xtb.fragment_unsupported(part, args)
            if reason:
                failures.append({'rank': c['rank'], 'reason': reason, 'fragment': part})
    return failures


def load_validation():
    raw = VAL.read_text().splitlines(keepends=True)
    if len(raw) != 21723:
        raise ValueError('Expected exactly 21723 official-validation lines')
    rows, seen, counts = [], set(), {}
    for path in SOURCES:
        count = 0
        for line in path.open():
            r = json.loads(line); i = r['source_row_id']; count += 1
            if i in seen or i not in range(len(raw)):
                raise ValueError('Duplicate or out-of-range validation identity')
            seen.add(i)
            if r['source_split'] != 'val' or r['sample_count'] != 64 or sum(r['metrics']) != 64:
                raise ValueError('Invalid validation provenance/sample count')
            if hashlib.sha256(raw[i].encode()).hexdigest() != r['source_line_sha256']:
                raise ValueError('Raw validation line hash mismatch')
            if raw[i].strip().split('|')[0] != r['reaction']:
                raise ValueError('Reaction alignment mismatch')
            source, target = r['reaction'].split('>>')
            if canonical(source) != r['reactant'] or canonical(target) != r['target']:
                raise ValueError('Input/reference normalization mismatch')
            candidates = r['candidates']
            if len(candidates) > 10 or [c['rank'] for c in candidates] != list(range(1, len(candidates)+1)):
                raise ValueError('Saved top-10 pool/order mismatch')
            if any(type(c['count']) is not int or c['count'] <= 0 for c in candidates) or any(a['count'] < b['count'] for a, b in zip(candidates, candidates[1:])):
                raise ValueError('Invalid original frequency order')
            if sum(c['count'] for c in candidates) > r['valid_conserved_samples'] or r['valid_conserved_samples'] != r['metrics'][0]+r['metrics'][1]:
                raise ValueError('Conserved sample accounting mismatch')
            if len({c['smiles'] for c in candidates}) != len(candidates):
                raise ValueError('Duplicate saved candidate')
            ranks = [c['rank'] for c in candidates if c['smiles'] == r['target']]
            if (r['target_rank'] is not None and r['target_rank'] <= 10) and ranks != [r['target_rank']]:
                raise ValueError('Saved reference rank mismatch')
            r['_source_file'] = str(path.relative_to(ROOT)); r['_raw_input'] = source
            rows.append(r)
        counts[str(path.relative_to(ROOT))] = count
    if seen != set(range(21723)):
        raise ValueError('Incomplete validation row identity coverage')
    return sorted(rows, key=lambda r: r['source_row_id']), counts


def identities_and_catalogs(rows):
    token, matrix = input_functions(ROOT/'utils/data_utils.py')
    parent = list(range(len(rows))); seen = {}; catalogs = defaultdict(Counter); witnesses = defaultdict(lambda: defaultdict(list))
    exact_catalogs = defaultdict(Counter)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, r in enumerate(rows):
        exact = model_input_hash(r['_raw_input'], token, matrix); r['_exact_input'] = exact
        normalized = r['reactant']; catalogs[normalized][r['target']] += 1
        exact_catalogs[exact][r['target']] += 1
        witnesses[normalized][r['target']].append(r['source_row_id'])
        keys = [('exact', exact), ('normalized', normalized), ('provided_group', r['split_group'])]
        if str(r['raw_sequence_id']).isdigit():
            keys.append(('network', r['raw_sequence_id']))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    minima = {}
    for i, r in enumerate(rows):
        k = root(i); minima[k] = min(minima.get(k, r['source_row_id']), r['source_row_id'])
    for i, r in enumerate(rows):
        r['_group'] = fingerprint(['validation_connected_group', minima[root(i)]])
    return catalogs, exact_catalogs, witnesses


def deterministic_sample(eligible, quota=25):
    rng = random.Random(SEED); choices = {}
    for stratum in STRATA:
        groups = defaultdict(list)
        for r in eligible:
            if r['stratum'] == stratum:
                groups[r['group']].append(r)
        keys = sorted(groups); rng.shuffle(keys)
        for key in keys:
            groups[key].sort(key=lambda r: r['source_row_id']); rng.shuffle(groups[key])
        choices[stratum] = [(key, groups[key][0]) for key in keys]
    used, selected, counters = set(), [], Counter()
    while True:
        added = False
        for stratum in STRATA:
            while counters[stratum] < quota and choices[stratum]:
                group, candidate = choices[stratum].pop(0)
                if group in used:
                    continue
                used.add(group); selected.append(candidate); counters[stratum] += 1; added = True
                break
        if not added:
            return selected


def energy_ranking(case, scores, near_tie=1.):
    """No frequency values or labels enter the sorting key; full-pool abstention."""
    candidates = case['candidates']; failures = []
    if len(scores) != len(candidates):
        failures.append('incomplete_candidate_accounting')
    protocols = {s.get('computation_protocol_sha256') for s in scores}
    if len(protocols) != 1 or None in protocols:
        failures.append('missing_or_incompatible_physical_protocol')
    for c, s in zip(candidates, scores):
        if s.get('smiles') != c['smiles'] or s.get('rank') != c['rank']:
            failures.append('candidate_identity_mismatch')
        e = s.get('energy_hartree')
        if s.get('status') != 'ok' or not isinstance(e, (int, float)) or isinstance(e, bool) or not math.isfinite(e):
            failures.append(s.get('reason', 'unreliable_candidate'))
    if failures:
        return {'status': 'unresolved', 'reasons': sorted(set(failures)), 'recovered': False,
            'reference_rank_before': case['target_rank'], 'reference_rank_after': None,
            'operational_rank': 1, 'candidates': scores}
    order = sorted(range(len(candidates)), key=lambda i: (scores[i]['energy_hartree'], candidates[i]['smiles']))
    energies = [s['energy_hartree'] for s in scores]; lowest = energies[order[0]]
    rank_by_index = {i: j+1 for j, i in enumerate(order)}
    target_index = next(i for i, c in enumerate(candidates) if c['smiles'] == case['target'])
    annotated = []
    for i, s in enumerate(scores):
        annotated.append(dict(s, energy_rank=rank_by_index[i],
            shared_energy_rank=1+sum(e < energies[i] for e in energies),
            delta_from_frequency_kcal_mol=(energies[i]-energies[0])*HARTREE_KCAL,
            delta_from_minimum_kcal_mol=(energies[i]-lowest)*HARTREE_KCAL))
    # The first-conformer-only alternative uses no reference labels and reports sampling sensitivity.
    first = [s.get('first_conformer_energy_hartree') for s in scores]
    sensitivity = None
    if all(isinstance(e, (int, float)) and math.isfinite(e) for e in first):
        first_order = sorted(range(len(first)), key=lambda i: (first[i], candidates[i]['smiles']))
        sensitivity = {'first_conformer_winner_rank': first_order[0]+1,
            'winner_changes': first_order[0] != order[0],
            'reference_rank_first_conformer': first_order.index(target_index)+1,
            'candidate_spreads_kcal_mol': [s.get('conformer_spread_kcal_mol') for s in scores]}
    chosen = candidates[order[0]]['smiles']; recorded = {a['smiles'] for a in case['recorded_alternatives']}
    return {'status': 'fully_scored', 'recovered': chosen == case['target'],
        'reference_rank_before': case['target_rank'], 'reference_rank_after': rank_by_index[target_index],
        'reference_shared_minimum': energies[target_index] == lowest,
        'chosen_original_rank': order[0]+1, 'chosen_smiles': chosen,
        'chosen_is_recorded_outcome': chosen in recorded,
        'chosen_is_other_recorded_outcome': chosen in recorded and chosen != case['target'],
        'winner_runner_up_gap_kcal_mol': (energies[order[1]]-lowest)*HARTREE_KCAL if len(order)>1 else None,
        'near_tie_original_ranks': [i+1 for i, e in enumerate(energies) if (e-lowest)*HARTREE_KCAL <= near_tie],
        'conformer_sensitivity': sensitivity, 'candidates': annotated}


def svg_structure(smiles):
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles, PARSER))
    drawer = rdMolDraw2D.MolDraw2DSVG(640, 260)
    drawer.DrawMolecule(mol); drawer.FinishDrawing()
    return drawer.GetDrawingText()


def case_table(out, cases, assessed=None):
    assessed = assessed or {}
    out.mkdir(exist_ok=True, parents=True)
    entries = ['<!doctype html><meta charset="utf-8"><title>FlowER energy-only pilot</title>',
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:100%;height:auto}</style>',
        '<h1>Frozen validation failure-recovery pilot</h1><p>Energy-only diagnostic. Pending energies are not zero energies or measured failures. Reference recovery is not chemical adjudication or overall accuracy improvement.</p>']
    for case in cases:
        i = case['source_row_id']; result = assessed.get(i, {})
        entries.append(f'<h2>Validation line {i+1} · {html.escape(case["stratum"])}</h2>')
        entries.append('<p>Solvent: '+html.escape(json.dumps(case['solvent']))+'</p>')
        entries.append('<details><summary>Complete input and all recorded validation alternatives</summary>')
        entries.append('<pre>'+html.escape(case['reactant'])+'</pre>'+svg_structure(case['reactant']))
        for a in case['recorded_alternatives']:
            entries.append('<p>Recorded count '+str(a['count'])+'; source lines '+html.escape(str([j+1 for j in a['source_row_ids']]))+'</p><pre>'+html.escape(a['smiles'])+'</pre>'+svg_structure(a['smiles']))
        entries.append('</details><table><tr><th>Original rank / count</th><th>Complete candidate structure</th><th>Energy and new rank</th></tr>')
        for c in case['candidates']:
            score = next((s for s in result.get('candidates', []) if s.get('rank') == c['rank']), None)
            state = json.dumps(score, indent=2) if score else 'Not run; energy/rank unavailable'
            entries.append('<tr><td>'+str(c['rank'])+' / '+str(c['count'])+(' · reference' if c['smiles']==case['target'] else '')+'</td><td>'+svg_structure(c['smiles'])+'<details><summary>Exact saved SMILES</summary><pre>'+html.escape(c['smiles'])+'</pre></details></td><td><pre>'+html.escape(state)+'</pre></td></tr>')
        entries.append('</table>')
    if not cases:
        entries.append('<p>No cases met all declared eligibility requirements. See exclusions.jsonl and preflight.json. No physical jobs are scheduled.</p>')
    (out/'cases.html').write_text('\n'.join(entries))


def prepare(out, solvent_rule):
    if (out/'manifest.json').exists():
        raise ValueError('Frozen manifest exists; use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    p = protocol(solvent_rule); save(out/'protocol.json', p)
    print('Verifying seven authorized validation sources', flush=True)
    rows, counts = load_validation()
    print('21723 identities verified; building exact/normalized/network groups', flush=True)
    catalog, exact_catalog, witnesses = identities_and_catalogs(rows)
    exclusions, reviewed, eligible, populations = [], [], [], Counter()
    for r in rows:
        if r['target_rank'] is None or not 2 <= r['target_rank'] <= 10:
            continue
        first = r['candidates'][0]['smiles']; labels = catalog[r['reactant']]
        stratum = STRATA[0] if first in labels else STRATA[1]; populations[stratum] += 1
        chemical, proof = chemical_eligibility(r)
        item = {'source_row_id': r['source_row_id'], 'group': r['_group'], 'stratum': stratum,
            'chemical_proof': proof, 'exclusions': []}
        if not chemical:
            item['exclusions'].append('not_verified_interpretable_proton_choice')
        else:
            failures = static_pool_check(r); solvent = solvent_context(r, solvent_rule)
            item.update(static_failures=failures, solvent=solvent)
            if failures: item['exclusions'].append('static_candidate_pool_unsupported')
            if solvent['solvent'] is None: item['exclusions'].append('unresolved_solvent_context')
            reviewed.append(item)
        if item['exclusions']:
            exclusions.append(item); continue
        alternatives = [{'smiles': s, 'count': n, 'source_row_ids': witnesses[r['reactant']][s]}
                        for s, n in sorted(labels.items())]
        eligible.append(dict(item, row_id=r['source_row_id'], reactant=r['reactant'], target=r['target'], target_rank=r['target_rank'],
            candidates=r['candidates'], candidate_pool_sha256=fingerprint(r['candidates']),
            recorded_alternatives=alternatives, exact_input_sha256=r['_exact_input'],
            top1_has_exact_input_recorded_match=first in exact_catalog[r['_exact_input']],
            source_file=r['_source_file'], source_line_sha256=r['source_line_sha256'],
            sample_count=r['sample_count'], network_id=r['raw_sequence_id']))
    selected = deterministic_sample(eligible)
    planned_jobs = sorted({(c['solvent']['solvent'], fragment)
        for c in selected for candidate in c['candidates'] for fragment in fragments(candidate['smiles'])})
    if len(planned_jobs) > p['budget']['max_unique_jobs']:
        raise ValueError('Selection exceeds fragment cap; amend protocol before any physical execution')
    jsonl(out/'selected-cases.jsonl', selected); jsonl(out/'exclusions.jsonl', exclusions)
    jsonl(out/'chemical-eligibility.jsonl', reviewed)
    save(out/'jobs.json', [{'solvent': solvent, 'fragment': part} for solvent, part in planned_jobs])
    summary = {}
    for stratum in STRATA:
        e = [r for r in eligible if r['stratum']==stratum]; s = [r for r in selected if r['stratum']==stratum]
        review = [r for r in reviewed if r['stratum']==stratum]
        reasons = Counter(reason for r in exclusions if r['stratum']==stratum for reason in r['exclusions'])
        summary[stratum] = {'recoverable_rows': populations[stratum], 'graph_eligible_rows': len(review),
            'graph_eligible_groups': len({r['group'] for r in review}),
            'physically_eligible_rows': len(e), 'available_groups': len({r['group'] for r in e}),
            'selected_rows': len(s), 'selected_groups': len({r['group'] for r in s}),
            'exclusions': dict(reasons), 'complete_pool_coverage': None, 'recovered': None,
            'recovery_all_selected': None, 'recovery_fully_scored': None}
    executable = shutil.which('xtb')
    preflight = {'status': 'prepared_physical_execution_held', 'validation_rows': len(rows),
        'unique_validation_rows': len({r['source_row_id'] for r in rows}), 'source_counts': counts,
        'cohorts': summary, 'solvent_rule': solvent_rule, 'selected_rows': len(selected),
        'candidate_entries': sum(len(c['candidates']) for c in selected),
        'unique_fragment_solvent_jobs_upper_bound': len(planned_jobs),
        'prescribed_conformer_optimizations_upper_bound': 2*len(planned_jobs),
        'job_wall_seconds_upper_bound': 20*len(planned_jobs),
        'two_worker_ideal_scheduling_seconds_upper_bound': 20*math.ceil(len(planned_jobs)/2),
        'hard_elapsed_cap_seconds': 7200, 'worker_thread_hours_cap': 4,
        'cost': {'new_cloud_spend': 0, 'physical_compute_performed': False,
                 'local_electricity_cost': 'not estimated; no physical run'},
        'xtb_executable': executable, 'binary_readiness': 'found_not_executed' if executable else 'not_found_on_PATH',
        'runtime_limit_note': 'job bound excludes orchestration; global elapsed deadline remains authoritative',
        'chemical_review': 'automated conservative graph proof; no independent chemical preference adjudication'}
    save(out/'preflight.json', preflight)
    sources = {str(path.relative_to(ROOT)): sha(path) for path in [VAL]+SOURCES}
    files = ['protocol.json', 'selected-cases.jsonl', 'exclusions.jsonl', 'chemical-eligibility.jsonl', 'jobs.json', 'preflight.json']
    code = {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), ROOT/'scripts/score_xtb_candidates.py', ROOT/'scripts/rerank_analysis.py', ROOT/'scripts/phase0_reranking_audit.py', ROOT/'scripts/audit_recorded_targets.py', ROOT/'utils/data_utils.py']}
    save(out/'manifest.json', {'status': 'frozen_before_energy_calculation', 'sources_sha256': sources,
        'code_sha256': code, 'files_sha256': {name: sha(out/name) for name in files},
        'rdkit_version': rdBase.rdkitVersion, 'selected_source_row_ids': [c['source_row_id'] for c in selected],
        'no_test_sources_opened': True, 'physical_execution_authorized': False})
    case_table(out, selected)
    print(json.dumps(preflight, indent=2), flush=True)


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text())
    for path, expected in manifest['sources_sha256'].items():
        if (ROOT/path).resolve() not in {p.resolve() for p in [VAL]+SOURCES}:
            raise ValueError('Forbidden data source in manifest')
        if sha(ROOT/path) != expected: raise ValueError('Source hash mismatch: '+path)
    for path, expected in manifest['code_sha256'].items():
        if sha(ROOT/path) != expected: raise ValueError('Code hash mismatch: '+path)
    for name, expected in manifest['files_sha256'].items():
        if sha(out/name) != expected: raise ValueError('Frozen artifact hash mismatch: '+name)
    return manifest


def summarize(out, score_file):
    verify(out)
    cases = [json.loads(line) for line in (out/'selected-cases.jsonl').open()]
    supplied = json.loads(score_file.read_text())
    if supplied['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('Scores belong to another frozen cohort')
    by_row = {int(k): v for k, v in supplied['cases'].items()}
    if set(by_row) != {c['source_row_id'] for c in cases}:
        raise ValueError('Every selected row must be accounted for, with no extra rows')
    assessed = {c['source_row_id']: energy_ranking(c, by_row[c['source_row_id']]) for c in cases}
    result = {'scope': 'selected validation failures only; no overall accuracy inference', 'strata': {}}
    for stratum in STRATA:
        ids = [c['source_row_id'] for c in cases if c['stratum']==stratum]
        complete = sum(assessed[i]['status']=='fully_scored' for i in ids)
        rescued = sum(assessed[i]['recovered'] for i in ids)
        result['strata'][stratum] = {'selected': len(ids), 'fully_scored': complete,
            'complete_pool_coverage': complete/len(ids) if ids else None,
            'references_promoted': rescued, 'unresolved': len(ids)-complete,
            'recovery_all_selected': rescued/len(ids) if ids else None,
            'recovery_fully_scored': rescued/complete if complete else None}
    result['runtime_and_cost'] = supplied.get('runtime_and_cost')
    result['independent_chemical_assessment'] = 'not performed'
    output = out/'assessment'
    if output.exists(): raise ValueError('Assessment already exists; no overwrite')
    output.mkdir(); save(output/'summary.json', result)
    jsonl(output/'case-results.jsonl', [dict(source_row_id=i, **v) for i, v in assessed.items()])
    case_table(output, cases, assessed)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'check', 'summarize', 'run'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--solvent-rule', choices=['documented', 'sole-organic-spectator'], default='documented')
    p.add_argument('--scores', type=Path)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.out, args.solvent_rule)
    elif args.command == 'check':
        verify(args.out); print(json.dumps({'frozen_hashes_verified': True, 'execution': 'held',
            'preflight': json.loads((args.out/'preflight.json').read_text())}, indent=2))
    elif args.command == 'summarize':
        if args.scores is None: raise SystemExit('--scores required')
        summarize(args.out, args.scores)
    else:
        raise SystemExit('Physical execution is held by the user. Explicit authorization and a reviewed frozen-protocol runner are required; no calculations started.')


if __name__ == '__main__':
    main()
