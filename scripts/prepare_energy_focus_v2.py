#!/usr/bin/env python3
"""Read-only physical audit and freeze for the same 18 cases; never runs xTB."""
import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

from rdkit import Chem

import energy_only_pilot as pilot
import run_energy_focus as focus
import score_xtb_candidates as old
import score_xtb_candidates_v2 as adapter
import xtb_structure_v2 as structure
from xtb_671_reference_state import BINARY, BINARY_SHA256

ROOT=pilot.ROOT
PARENT=ROOT/'results/reranking/energy-only-focus-20261031'
PROPOSAL=ROOT/'results/reranking/energy-only-frozen-solvents-20261031'


def tree_hashes(path):
    return {str(p.relative_to(path)):pilot.sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def settings(solvent,cache):
    return SimpleNamespace(xtb=str(BINARY),xtb_sha256=BINARY_SHA256,solvent=solvent,
        max_heavy_atoms=40,strict_elements=True,cache=cache)


def job_id(solvent,smiles):
    return pilot.fingerprint({'solvent':solvent,'smiles':smiles})


def verify_freeze(prepared):
    manifest=json.loads((prepared/'manifest.json').read_text())
    for name,h in manifest['code_sha256'].items():
        if pilot.sha(ROOT/name)!=h:raise ValueError('Frozen code changed: '+name)
    for name,h in manifest['artifact_sha256'].items():
        if pilot.sha(prepared/name)!=h:raise ValueError('Frozen artifact changed: '+name)
    if tree_hashes(Path(manifest['parent_directory']))!=manifest['original_tree_sha256']:
        raise ValueError('Original run changed')
    if pilot.sha(BINARY)!=BINARY_SHA256 or BINARY_SHA256!=manifest['xtb_binary_sha256']:
        raise ValueError('Inspected xTB backend changed')
    frozen_protocol=json.loads((prepared/'protocol.json').read_text())
    for solvent,expected in frozen_protocol['per_solvent_computation_protocol'].items():
        if adapter.protocol(settings(solvent,prepared/'validated-cache'))!=expected:
            raise ValueError('Runtime physical protocol or dependency version changed')
    cases,_,_=focus.load_focus(PROPOSAL)
    if [json.loads(l) for l in (prepared/'selected-cases.jsonl').read_text().splitlines()]!=cases:
        raise ValueError('Frozen cases, candidates or solvents changed')
    return manifest,cases


def prepare(out):
    if out.exists():raise ValueError('Use a fresh preparation directory')
    original=tree_hashes(PARENT)
    cases,parent_protocol,_=focus.load_focus(PROPOSAL)
    if cases!=[json.loads(l) for l in (PARENT/'selected-cases.jsonl').read_text().splitlines()]:
        raise ValueError('Original selected cases differ from proposal')
    completion=json.loads((PARENT/'completion.json').read_text())
    for name,h in completion['output_sha256'].items():
        if pilot.sha(PARENT/name)!=h:raise ValueError('Original completion hash mismatch')
    if pilot.sha(BINARY)!=BINARY_SHA256:raise ValueError('Backend changed')
    execution=json.loads((PARENT/'execution.json').read_text())
    # These full per-solvent score artifacts were hashed at original completion.
    # Cross-check caches against them, not just against new hashes made today.
    published={}
    for score_file in sorted(PARENT.glob('scores-*.jsonl')):
        for line in score_file.read_text().splitlines():
            for fragment in json.loads(line).get('fragments',[]):
                key=(fragment['solvent'],fragment['smiles'])
                if key in published and published[key]!=fragment:
                    raise ValueError('Conflicting original calculation provenance')
                published[key]=fragment
    out.mkdir(parents=True);(out/'validated-cache').mkdir()
    shutil.copyfile(PARENT/'selected-cases.jsonl',out/'selected-cases.jsonl')
    expected={(c['solvent']['solvent'],f) for c in cases for candidate in c['candidates'] for f in pilot.fragments(candidate['smiles'])}
    records={}
    for path in sorted((PARENT/'cache').glob('*.json')):
        d=json.loads(path.read_text());key=(d['solvent'],d['smiles'])
        if key in records:raise ValueError('Duplicate fragment cache')
        if published.get(key)!=d:raise ValueError('Cache differs from original completion-hashed calculation')
        records[key]=(path,d)
    if set(records)!=expected or len(expected)!=111:raise ValueError('Fragment plan changed')
    diagnostics=[];plan=[];imported=set();old_failures=[]
    for (solvent,smiles),(path,d) in sorted(records.items()):
        decision=adapter.reuse_decision(d,execution)
        item={'job_id':job_id(solvent,smiles),'solvent':solvent,'smiles':smiles,
            'source_cache':str(path.relative_to(ROOT)),'source_sha256':pilot.sha(path),
            'original_status':d['status'],'reuse_decision':decision,
            'calculation_kind':'monatomic_single_point' if structure.expected_molecule(smiles).GetNumAtoms()==1 else 'molecular_optimization'}
        if decision['reusable']:
            data=adapter.imported_result(d,decision,settings(solvent,out/'validated-cache'),path,pilot.sha(path))
            dest=old.cache_path(adapter.configuration(smiles,settings(solvent,out/'validated-cache')),settings(solvent,out/'validated-cache'))
            old.atomic_json(dest,data)
            item.update(action='reuse_validated_success',imported_cache=str(dest.relative_to(out)))
            imported.add((solvent,smiles))
        else:
            item['action']='new_calculation_required'
        plan.append(item)
        for c in d.get('conformer_records',[]):
            diag={'job_id':item['job_id'],'source_cache':item['source_cache'],'source_sha256':item['source_sha256'],
                'solvent':solvent,'smiles':smiles,'conformer_id':c['conformer_id'],
                'original_fragment_status':d['status'],'original_conformer_status':c['status'],
                'original_reason':c.get('reason'),'returncode':c.get('returncode'),
                'energy_reused':decision['reusable'],'optimized_geometry_available':'optimized_xyz' in c}
            if 'optimized_xyz' in c:
                diag['structure_v2']=structure.check_geometry(smiles,c['input_xyz'],c['optimized_xyz'])
                if c['status']!='ok':
                    diag['input_geometry_diagnosis']=structure.check_geometry(smiles,c['input_xyz'],c['input_xyz'])
                    old_failures.append(diag)
            else:diag['structure_v2']={'valid':False,'reason':'no_saved_converged_geometry'}
            diagnostics.append(diag)
    coverage=[]
    for c in cases:
        entries=[]
        for candidate in c['candidates']:
            fs=pilot.fragments(candidate['smiles']);solvent=c['solvent']['solvent']
            entries.append({'original_rank':candidate['rank'],'fragment_occurrences':len(fs),
                'reusable_fragment_occurrences':sum((solvent,f) in imported for f in fs),
                'fully_available_from_validated_cache':all((solvent,f) in imported for f in fs),
                'pending_job_ids':sorted({job_id(solvent,f) for f in fs if (solvent,f) not in imported})})
        coverage.append({'source_row_id':c['source_row_id'],'validation_line':c['source_row_id']+1,
            'solvent':c['solvent']['solvent'],'candidate_count':len(entries),'candidates':entries,
            'complete_pool_from_cache':all(e['fully_available_from_validated_cache'] for e in entries)})
    pending=[p for p in plan if p['action']=='new_calculation_required']
    kinds=Counter(p['calculation_kind'] for p in pending)
    diag_counts=Counter(d['structure_v2']['reason'] for d in old_failures)
    rule={'version':adapter.VERSION,'execution_status':'PREPARED_PHYSICAL_EXECUTION_HELD',
        'parent_protocol':parent_protocol,'changes':{
            'monatomic_ions':'one GFN2-xTB/ALPB single point at origin; fixed formal charge, UHF=0; no geometry optimization or fictitious duplicate conformers',
            'H2':'RDKit covalent-radii connectivity (useVdw=True, covFactor=1.3) only for two-H fragments; H-H bond must be reconstructed',
            'molecular_structure':'independent indexed full graph including hydrogen attachments; preserve specified stereo; bond-order reconstruction plus capped indexed resonance proof; no template bond assignment',
            'unresolved_valence':'PF6/nitro reconstruction errors and unproved sulfur representations remain unresolved; no exception by molecule or reference label',
            'reuse':'only previously successful complete fragments; recheck physical settings, command/output, SCC/optimizer termination, XYZ energy provenance, every conformer structure; no energies salvaged from failed fragments',
            'reference_state_spelling':'bar1M directly to same hash-bound xTB 6.7.1 backend; same 1-bar gas/1-M solution convention'},
        'per_solvent_computation_protocol':{s:adapter.protocol(settings(s,out/'validated-cache')) for s in sorted({c['solvent']['solvent'] for c in cases})},
        'candidate_and_solvent_policy':'unchanged byte-for-byte selected-cases.jsonl; all 62 candidates; no new selection or reference-dependent checks',
        'pending_jobs_policy':'one new attempt per pending fragment; molecules retain original two conformers, seed42, normal optimization, 20-second whole-fragment cap; no adaptive retries',
        'budget':{'workers':2,'threads_per_worker':1,'maximum_elapsed_seconds':7200,'fragment_wall_seconds':20,
            'cloud_rental':False,'training':False,'new_flowER_inference':False},
        'fallback':'any unavailable/unreliable candidate makes whole row unresolved; retain frequency winner and all 18 denominator rows',
        'reporting':'calculation coverage separate from reference recovery; no revised recovery calculated during preparation',
        'authorization':'fresh explicit authorization receipt bound to this manifest required before physical execution'}
    worker_bound=math.ceil(len(pending)/2)*20
    summary={'version':adapter.VERSION,'physical_execution_held':True,'selected_cases':18,'candidate_entries':62,
        'fragment_solvent_jobs':111,'validated_reusable_jobs':len(imported),'pending_jobs':len(pending),
        'pending_job_types':dict(kinds),'maximum_new_xtb_calls':kinds['monatomic_single_point']+2*kinds['molecular_optimization'],
        'coverage_before_rerun':{'fully_cached_candidates':sum(e['fully_available_from_validated_cache'] for r in coverage for e in r['candidates']),
            'complete_pools':sum(r['complete_pool_from_cache'] for r in coverage),'selected_pools':18},
        'reference_recovery_v2':None,'reference_recovery_note':'not assessed; no revised physical run or ranking performed',
        'saved_failed_conformers_with_geometry':len(old_failures),'structure_diagnosis_counts':dict(diag_counts),
        'cost_preflight':{'planning_allowance_elapsed_seconds':worker_bound+80,'worker_schedule_bound_seconds_excluding_controller_overhead':worker_bound,
            'total_fragment_worker_seconds_cap':len(pending)*20,'basis':'fixed 20-second fragment cap, at most two simultaneous workers; add 80 seconds for setup/reporting; planning allowance, not a measured runtime or success forecast',
            'hard_total_elapsed_cap_seconds':7200,'new_cloud_cost_usd':0,'local_electricity_cost':'not measured'}}
    pilot.save(out/'protocol.json',rule);pilot.save(out/'summary.json',summary)
    pilot.jsonl(out/'fragment-plan.jsonl',plan);pilot.jsonl(out/'geometry-diagnostics.jsonl',diagnostics)
    pilot.jsonl(out/'case-coverage.jsonl',coverage)
    write_report(out,summary,coverage,old_failures)
    code=[Path(__file__),Path(adapter.__file__),Path(structure.__file__),ROOT/'scripts/run_energy_focus_v2.py',
        Path(focus.__file__),Path(old.__file__),Path(pilot.__file__),ROOT/'scripts/rerank_analysis.py',
        ROOT/'scripts/xtb_671_reference_state.py',
        ROOT/'tests/test_xtb_correction_v2.py']
    if tree_hashes(PARENT)!=original:raise RuntimeError('Original run changed during preparation')
    manifest={'version':adapter.VERSION,'freeze_revision':2,'status':'FROZEN_AWAITING_EXPLICIT_PHYSICAL_AUTHORIZATION',
        'parent_directory':str(PARENT),'original_tree_sha256':original,
        'code_sha256':{str(p.relative_to(ROOT)):pilot.sha(p) for p in code},
        'artifact_sha256':tree_hashes(out),'xtb_binary_sha256':BINARY_SHA256,
        'selected_cases_sha256':pilot.sha(out/'selected-cases.jsonl')}
    pilot.save(out/'manifest.json',manifest)
    verify_freeze(out)
    print(json.dumps(summary,indent=2))


def write_report(out,s,coverage,diagnostics):
    lines=['# Frozen v2 correction — preparation only','',
        '**No new physical calculations have run. Original results are unchanged.**','',
        f"Same 18 cases, 62 candidates and 111 fragment–solvent jobs. **{s['validated_reusable_jobs']} previously successful jobs passed revalidation; {s['pending_jobs']} require a new calculation.** No energy is imported from an originally failed fragment, including its successful individual conformers.",'',
        '## Calculation coverage — independent of reference recovery','',
        f"Validated cache currently covers **{s['coverage_before_rerun']['fully_cached_candidates']}/62 candidates** and **{s['coverage_before_rerun']['complete_pools']}/18 complete pools**. This is availability of reliable calculations, not reference recovery. V2 reference recovery has not been assessed. Successful revalidation of an old geometry does not turn an old failed calculation into an accepted energy.",'',
        '## Revised rules','',
        '- Monatomic ions: one fresh single point with the declared charge, spin, solvent and reference state. Require SCC convergence, normal termination and finite total energy. No optimization and no duplicate conformers.',
        '- Molecules: preserve the two prescribed conformers, seed, optimizer and timeout. Check ordered atom inventory, all indexed bonds including hydrogen attachments, specified stereochemistry, bond orders and charges. H₂ uses covalent-radius connectivity; other molecules retain the original connectivity method.',
        '- Resonance equivalence must be demonstrated by a shared indexed resonance form, with a 256-form cap. Connectivity alone never suffices. Reconstruction exceptions and unproved identities remain unresolved; no reference-derived mapping or template bond-order replacement.',
        '- Whole-row fallback, complete candidate accounting, fixed solvents and energy-only ranking remain unchanged. No failed calculation energies are salvaged and no case is replaced.','',
        '## Saved-geometry diagnosis','',
        '| Diagnosis of previously rejected conformers with saved optimized XYZ | Count |','| --- | ---: |']
    lines += [f'| {k} | {v} |' for k,v in s['structure_diagnosis_counts'].items()]
    lines += ['', 'The connectivity flags retain atom indices and changed-bond distances; they are not all claimed to be proven chemical transformations. H₂ recognition and proven resonance matches are postprocessing explanations only. PF₆⁻/nitro reconstruction failures and unproved sulfur representations stay unresolved. See the linked audit for input/output comparisons and source cache hashes.','',
        '## Rerun cost, before execution','',
        f"Plan: **{s['pending_job_types'].get('monatomic_single_point',0)} monatomic single-point jobs + {s['pending_job_types'].get('molecular_optimization',0)} two-conformer molecular jobs**, at most **{s['maximum_new_xtb_calls']} xTB calls**. Allow **{s['cost_preflight']['planning_allowance_elapsed_seconds']/60:.1f} minutes locally**, including a conservative {s['cost_preflight']['worker_schedule_bound_seconds_excluding_controller_overhead']/60:.1f}-minute bounded-worker schedule plus setup/reporting. Two workers, one thread each; unchanged two-hour absolute cap. Cloud cost **$0**; local electricity unmeasured. This is a scheduling estimate, not a forecast of successful coverage. Unchanged molecular optimizations may repeat their previous failures.",'',
        '## Per-case calculation availability','',
        '| Validation line | Frozen solvent | Candidates reusable in full | Complete pool |','| --- | --- | ---: | --- |']
    lines += [f"| {r['validation_line']} | {r['solvent']} | {sum(c['fully_available_from_validated_cache'] for c in r['candidates'])}/{r['candidate_count']} | {r['complete_pool_from_cache']} |" for r in coverage]
    lines += ['', '[Frozen protocol](protocol.json) · [Manifest](manifest.json) · [Geometry audit](geometry-review.html) · [Job plan](fragment-plan.jsonl) · [Machine-readable coverage](case-coverage.jsonl)','',
        'Preparation can be reproduced into a new directory with:','',
        '```bash','.venv-analysis/bin/python scripts/prepare_energy_focus_v2.py --out results/reranking/energy-only-focus-v2-20261031','```','',
        'Execution remains held. After explicit authorization, supply a receipt with `local_physical_execution_authorized: true`, the exact `manifest_sha256`, `workers: 2`, and `maximum_elapsed_seconds: 7200`:','',
        '```bash','.venv-analysis/bin/python scripts/run_energy_focus_v2.py \\',
        '  --prepared results/reranking/energy-only-focus-v2-20261031 \\',
        '  --out results/reranking/energy-only-focus-v2-20261031-run \\',
        '  --authorization /path/to/new-v2-authorization.json','```','',
        'Method references: [xTB single points](https://xtb-docs.readthedocs.io/en/latest/sp.html), [xTB optimization convergence](https://xtb-docs.readthedocs.io/en/latest/optimization.html), [RDKit bond reconstruction](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Saved geometry diagnosis</title>',
        '<style>body{font:15px system-ui;margin:2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:10px;border:1px solid #ccc;vertical-align:top}table{border-collapse:collapse;width:100%}</style>',
        '<h1>Previously rejected geometries — v2 audit only</h1><p>No failed-calculation energy is accepted. No energy or reference outcome is used by the structural checker.</p>',
        '<table><tr><th>Fragment / saved calculation</th><th>Diagnosis</th></tr>']
    for d in diagnostics:
        page.append('<tr><td><pre>'+html.escape(d['smiles'])+'</pre>'+html.escape(d['solvent'])+
            ' · conformer '+str(d['conformer_id'])+'<pre>'+html.escape(d['source_cache'])+'</pre></td><td><pre>'+
            html.escape(json.dumps({k:v for k,v in d.items() if k not in ('smiles','source_cache')},indent=2))+'</pre></td></tr>')
    page.append('</table>');(out/'geometry-review.html').write_text('\n'.join(page))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.out)
