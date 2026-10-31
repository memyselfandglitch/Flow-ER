#!/usr/bin/env python3
"""Audit completed V2 artifacts and report outcomes; never execute calculations."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import energy_only_pilot as pilot
import prepare_energy_focus_v2 as preparation
import score_xtb_candidates as old
import score_xtb_candidates_v2 as adapter
import xtb_structure_v2 as structure


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def report(prepared,run):
    manifest,cases=preparation.verify_freeze(prepared)
    completion=json.loads((run/'completion.json').read_text())
    assert completion['status']=='complete'
    for name,h in completion['output_sha256'].items():
        assert pilot.sha(run/name)==h, name
    assert completion['prepared_manifest_sha256']==pilot.sha(prepared/'manifest.json')
    assert (run/'selected-cases.jsonl').read_bytes()==(prepared/'selected-cases.jsonl').read_bytes()
    source_summary=json.loads((run/'summary.json').read_text())
    results=read_rows(run/'case-results.jsonl');by_row={r['source_row_id']:r for r in results}
    plan=read_rows(prepared/'fragment-plan.jsonl')
    cached={}
    for p in (run/'cache').glob('*.json'):
        d=json.loads(p.read_text());key=(d['solvent'],d['smiles'])
        assert key not in cached
        cached[key]=d
    assert len(cached)==111
    reused=0;new_status=Counter();accepted_conformers=0;monatomic_status=Counter();job_explanations={}
    for job in plan:
        args=preparation.settings(job['solvent'],run/'cache')
        config=adapter.configuration(job['smiles'],args)
        path=old.cache_path(config,args);d=json.loads(path.read_text())
        assert all(d[k]==v for k,v in config.items())
        if job['action']=='reuse_validated_success':
            assert path.read_bytes()==(prepared/job['imported_cache']).read_bytes()
            assert d['status']=='ok';reused+=1
        else:
            assert 'reuse' not in d;new_status[d['status']]+=1
            if structure.expected_molecule(d['smiles']).GetNumAtoms()==1:monatomic_status[d['status']]+=1
        explanations=[]
        if 'UnicodeDecodeError' in d.get('stderr',''):
            explanations.append('xTB output could not be decoded as UTF-8; worker failed and no energy was accepted')
        if any('FAILED TO CONVERGE GEOMETRY OPTIMIZATION' in c.get('stdout','').upper() for c in d.get('conformer_records',[])):
            explanations.append('geometry optimization explicitly failed to converge despite exit code 0/normal program termination')
        if explanations:job_explanations[(d['solvent'],d['smiles'])]=explanations
        if d['status']=='ok':
            mono=structure.expected_molecule(d['smiles']).GetNumAtoms()==1
            need=1 if mono else 2
            assert len(d['conformer_records'])==need and d['complete_conformers']
            es=[]
            for c in d['conformer_records']:
                v=adapter.validate_conformer(d['smiles'],c,d['solvent'],mono)
                assert c['status']=='ok' and v['valid'],(job['job_id'],v)
                assert v['energy_hartree']==c['energy_hartree'];es.append(v['energy_hartree'])
                accepted_conformers+=1
            assert es==d['conformer_energies_hartree'] and min(es)==d['energy_hartree']
        else:assert d['energy_hartree'] is None
    assert reused==71 and sum(new_status.values())==40
    assert len(by_row)==len(cases)==18 and len({c['group'] for c in cases})==18
    outcomes=Counter();rows=[];failure_counts=Counter();candidate_count=0;scored_candidates=0
    for case in cases:
        r=by_row[case['source_row_id']];scores=r['candidates'];candidate_count+=len(scores)
        assert case['candidates'][0]['smiles']!=case['target']
        assert [(s['rank'],s['smiles']) for s in scores]==[(c['rank'],c['smiles']) for c in case['candidates']]
        for s in scores:
            fragments=[cached[(case['solvent']['solvent'],f)] for f in pilot.fragments(s['smiles'])]
            available=all(d['status']=='ok' for d in fragments)
            assert (s['status']=='ok')==available
            if available:
                assert math.isclose(s['energy_hartree'],sum(d['energy_hartree'] for d in fragments),abs_tol=1e-10)
                scored_candidates+=1
            else:assert s['energy_hartree'] is None
        full=all(s['status']=='ok' for s in scores)
        assert (r['status']=='fully_scored')==full
        row={'source_row_id':case['source_row_id'],'validation_line':r['validation_line'],
            'assumed_solvent':r['nominal_solvent'],'candidate_count':len(scores),
            'reference_original_rank':r['reference_rank_before'],'reference_energy_rank':r['reference_rank_after'],
            'chosen_original_rank':r.get('chosen_original_rank'),
            'energy_gap_to_runner_up_kcal_mol':r.get('winner_runner_up_gap_kcal_mol'),
            'near_tie_original_ranks':r.get('near_tie_original_ranks',[]),
            'conformer_sensitivity':r.get('conformer_sensitivity'),
            'failure_reasons':r.get('failure_details',[]),
            'chosen_is_other_recorded_outcome':r.get('chosen_is_other_recorded_outcome'),
            'failure_explanations':sorted({why for c in case['candidates'] for f in pilot.fragments(c['smiles'])
                for why in job_explanations.get((case['solvent']['solvent'],f),[])}),
            'winner_delta_vs_frequency_kcal_mol':None,'reference_gap_above_winner_kcal_mol':None}
        if full:
            ordered=sorted(scores,key=lambda s:(s['energy_hartree'],s['smiles']))
            winner=ordered[0];ref=next(s for s in scores if s['smiles']==case['target'])
            assert winner['rank']==r['chosen_original_rank']
            assert r['recovered']==(winner['smiles']==case['target'])
            assert r['reference_rank_after']==next(i+1 for i,s in enumerate(ordered) if s['smiles']==case['target'])
            row['winner_delta_vs_frequency_kcal_mol']=(winner['energy_hartree']-scores[0]['energy_hartree'])*pilot.HARTREE_KCAL
            row['reference_gap_above_winner_kcal_mol']=(ref['energy_hartree']-winner['energy_hartree'])*pilot.HARTREE_KCAL
            row['exact_minimum_tie_count']=sum(s['energy_hartree']==winner['energy_hartree'] for s in scores)
            outcome='reference_promoted' if r['recovered'] else ('unchanged_energy_winner' if winner['rank']==1 else 'switch_to_other_strictly_incorrect_candidate')
        else:
            outcome='unresolved_frequency_fallback'
            assert not r['recovered'] and r['reference_rank_after'] is None
            failure_counts.update(set(row['failure_reasons']))
        row['outcome']=outcome;outcomes[outcome]+=1;rows.append(row)
    assert candidate_count==62 and sum(outcomes.values())==18
    full=18-outcomes['unresolved_frequency_fallback'];recover=outcomes['reference_promoted']
    assert source_summary['calculation_coverage']['complete_pools']==full
    assert source_summary['reference_recovery']['references_promoted_to_energy_rank1']==recover
    v1=read_rows(preparation.PARENT/'case-results.jsonl')
    previous_recovered={r['source_row_id'] for r in v1 if r['recovered']}
    current_recovered={r['source_row_id'] for r in rows if r['outcome']=='reference_promoted'}
    summary={'coverage':{'complete_pools':full,'selected_pools':18,'complete_pool_fraction':full/18,
        'fully_scored_candidates':scored_candidates,'candidate_entries':62,'reused_validated_jobs':reused,
        'new_job_status_counts':dict(new_status),'new_monatomic_job_status_counts':dict(monatomic_status)},
        'ranking_outcomes':{k:outcomes[k] for k in ['reference_promoted','unchanged_energy_winner',
            'switch_to_other_strictly_incorrect_candidate','unresolved_frequency_fallback']},
        'operationally_unchanged_including_fallback':outcomes['unchanged_energy_winner']+outcomes['unresolved_frequency_fallback'],
        'recovery_all_selected':recover/18,'recovery_fully_scored':recover/full if full else None,
        'near_tie_definition_kcal_mol':1.,'near_tie_complete_pools':sum(len(r['near_tie_original_ranks'])>1 for r in rows),
        'exact_minimum_tie_complete_pools':sum(r.get('exact_minimum_tie_count',0)>1 for r in rows),
        'conformer_winner_changes':sum(bool(r['conformer_sensitivity'] and r['conformer_sensitivity']['winner_changes']) for r in rows),
        'switches_to_another_recorded_outcome':sum(r['chosen_is_other_recorded_outcome'] is True for r in rows),
        'failure_reasons_by_unresolved_row_overlapping':dict(failure_counts),
        'runtime_and_cost':source_summary['runtime_and_cost'],
        'comparison_with_preserved_v1':{'v1_complete_pools':sum(r['status']=='fully_scored' for r in v1),
            'v1_recovered_references':len(previous_recovered),
            'additional_recovered_validation_lines':[i+1 for i in sorted(current_recovered-previous_recovered)],
            'previous_recoveries_not_retained_validation_lines':[i+1 for i in sorted(previous_recovered-current_recovered)]},
        'interpretation':'Failure-only diagnostic; strict reference recovery is not independent chemical assessment or overall accuracy gain',
        'remaining_holds':['DeepMech','training','new FlowER inference','cloud execution','reserved-test outcomes']}
    pilot.save(run/'assessment-summary.json',summary);pilot.jsonl(run/'assessment-cases.jsonl',rows)
    write_report(run,summary,rows)
    verification={'status':'passed','original_run_unchanged':True,'frozen_preparation_unchanged':True,
        'candidate_pools_and_solvents_byte_identical':True,'validated_jobs_reused_byte_identically':71,
        'new_fragment_jobs':40,'accepted_conformers_revalidated':accepted_conformers,
        'full_pool_accounting_and_ranking_recomputed':True,'original_completion_hashes_verified':True,
        'analysis_code_sha256':pilot.sha(Path(__file__)),
        'new_report_sha256':{name:pilot.sha(run/name) for name in ('assessment-summary.json','assessment-cases.jsonl','DETAILED_REPORT.md')}}
    pilot.save(run/'assessment-verification.json',verification)
    print(json.dumps(summary,indent=2))


def write_report(run,s,rows):
    n=s['coverage']['complete_pools'];o=s['ranking_outcomes'];k=o['reference_promoted'];t=s['runtime_and_cost']['elapsed_seconds']
    lines=['# Frozen V2 execution — detailed assessment','',
        f"**Calculation coverage: {n}/18 fully scored pools, {s['coverage']['fully_scored_candidates']}/62 candidates.**",'',
        f"**Reference recovery: {k}/18 selected cases ({100*k/18:.1f}%) and {k}/{n} fully scored pools**"+(' (undefined denominator).' if not n else f" ({100*k/n:.1f}%)."),'',
        '| Mutually exclusive outcome | Cases |','| --- | ---: |',
        f'| Reference promoted to rank 1 | {k} |',
        f'| Fully scored; original frequency winner remains lowest-energy | {o["unchanged_energy_winner"]} |',
        f'| Fully scored; switch to another strictly incorrect candidate | {o["switch_to_other_strictly_incorrect_candidate"]} |',
        f'| Unresolved; original prediction retained by fallback | {o["unresolved_frequency_fallback"]} |','',
        f"Operationally, {s['operationally_unchanged_including_fallback']} predictions are unchanged including fallbacks. An unresolved fallback is not evidence that energy favors the frequency winner. Winners matching another recorded validation outcome: {s['switches_to_another_recorded_outcome']}.",'',
        f"Compared with preserved V1, the complete-pool count rises from {s['comparison_with_preserved_v1']['v1_complete_pools']} to {n}. The {k} recoveries include {s['comparison_with_preserved_v1']['v1_recovered_references']} from V1; additional recovered validation lines are {s['comparison_with_preserved_v1']['additional_recovered_validation_lines']}.",'',
        '## Complete-pool energy ranking','',
        'Energy gaps are in kcal/mol. ΔE versus frequency is the chosen energy winner minus the frequency winner. Near-ties use the frozen 1 kcal/mol band, for reporting only.','',
        '| Validation line | Assumed solvent | Outcome | Chosen original rank | Reference rank: original → energy | ΔE vs frequency | Gap to runner-up | Near-tie original ranks | First-conformer winner original rank |',
        '| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |']
    for r in rows:
        if r['outcome']=='unresolved_frequency_fallback':continue
        first=(r['conformer_sensitivity'] or {}).get('first_conformer_winner_rank')
        lines.append(f"| {r['validation_line']} | {r['assumed_solvent']} | {r['outcome']} | {r['chosen_original_rank']} | {r['reference_original_rank']} → {r['reference_energy_rank']} | {r['winner_delta_vs_frequency_kcal_mol']:.3f} | {r['energy_gap_to_runner_up_kcal_mol']:.3f} | {r['near_tie_original_ranks']} | {first} |")
    lines += ['',f"Near-ties in {s['near_tie_complete_pools']}/{n} fully scored pools; exact minimum ties in {s['exact_minimum_tie_complete_pools']}. Using only the first prescribed conformer changes the winner in {s['conformer_winner_changes']}/{n} complete pools. Per-candidate conformer ranges and reference energy gaps are retained in the detailed artifacts.",'',
        '## Unresolved cases','',
        '| Validation line | Assumed solvent | Reference original rank | Failure reasons |',
        '| --- | --- | ---: | --- |']
    for r in rows:
        if r['outcome']=='unresolved_frequency_fallback':
            lines.append(f"| {r['validation_line']} | {r['assumed_solvent']} | {r['reference_original_rank']} | {', '.join(r['failure_reasons'])} |")
    lines += ['', 'Failure counts below overlap; one case may contain multiple failed fragments. The generic incomplete-conformer-set reason is not an additional independent failure.','']
    lines += [f'- {reason}: {count} unresolved cases' for reason,count in s['failure_reasons_by_unresolved_row_overlapping'].items()]
    lines += ['', 'Additional diagnosis from the saved logs (no recalculation):','']
    for r in rows:
        for explanation in r['failure_explanations']:
            lines.append(f"- Validation line {r['validation_line']}: {explanation}.")
    lines += ['', '## Execution and integrity','',
        f"Reused 71 validated jobs unchanged. Executed 40 remaining jobs: {s['coverage']['new_job_status_counts']}. Scoring and aggregation took **{t:.1f} seconds ({t/60:.2f} minutes)**; OS-reported CPU time was {s['runtime_and_cost']['cpu_seconds']:.1f} seconds. Two local workers; **$0 cloud cost**. No automatic retry or post-result protocol adjustment occurred.",'',
        f"Monatomic single-point job outcomes: {s['coverage']['new_monatomic_job_status_counts']}. These are new, converged calculations; failed V1 ion energies were not reused.",'',
        'Original V1 results and the frozen V2 preparation were hash-verified unchanged. All 18 cases, 62 candidates, counts, original ranks and solvent assignments were retained. Every accepted conformer was revalidated against convergence, physical settings and structure rules. Failed calculations supplied no accepted energy; incomplete pools were never partially ranked.','',
        'DeepMech, training, new FlowER inference, cloud execution and reserved-test access remain held. This selected failure-only sample cannot measure damage to correct baseline predictions or establish overall accuracy gain. Approximate fragment-sum energies are not barriers, rates or complete Gibbs free energies.','',
        '[All candidate structures, energies, original ranks/counts and calculation provenance](cases.html) · [Summary](assessment-summary.json) · [Per-case assessment](assessment-cases.jsonl) · [Integrity receipt](assessment-verification.json) · [Execution log](run.log)']
    (run/'DETAILED_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();report(a.prepared,a.run)
