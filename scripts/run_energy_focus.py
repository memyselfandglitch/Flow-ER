#!/usr/bin/env python3
"""Execute the explicitly authorized 18-case local energy-only focus run."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

import energy_only_pilot as pilot
import score_xtb_candidates as xtb

ROOT = pilot.ROOT
FOCUS = 'no_recorded_outcome_match'


def load_focus(proposal):
    manifest = json.loads((proposal/'manifest.json').read_text())
    allowed = {str(p.relative_to(ROOT)) for p in [pilot.VAL]+pilot.SOURCES}
    if set(manifest['validation_source_sha256']) != allowed:
        raise ValueError('Unexpected source scope')
    for field, base in [('validation_source_sha256', ROOT), ('code_sha256', ROOT), ('artifact_sha256', proposal)]:
        for name, expected in manifest[field].items():
            if pilot.sha(base/name) != expected: raise ValueError('Frozen provenance changed: '+name)
    solvent_dir = ROOT/manifest['frozen_solvent_directory']
    for name, expected in manifest['frozen_solvent_files_sha256'].items():
        if pilot.sha(solvent_dir/name) != expected: raise ValueError('Frozen solvent assignment changed')
    all_cases = [json.loads(line) for line in (proposal/'selected-cases.jsonl').open()]
    cases = [c for c in all_cases if c['stratum']==FOCUS]
    if len(cases)!=18 or len({c['group'] for c in cases})!=18:
        raise ValueError('Expected the frozen 18 distinct unmatched groups')
    for c in cases:
        if pilot.fingerprint(c['candidates'])!=c['candidate_pool_sha256']:
            raise ValueError('Frozen pool changed')
        a=c['frozen_solvent_assignment']
        if a['status']!='assumed' or a['proposed_alpb_name']!=c['solvent']['solvent']:
            raise ValueError('Unresolved or changed solvent')
    return cases, json.loads((proposal/'protocol.json').read_text()), manifest


def physical_records(cases):
    """Do not pass target labels, correctness, counts or rarity to the scorer."""
    return [{'row_id':c['source_row_id'],'reactant':c['reactant'],
        'candidates':[{'rank':x['rank'],'smiles':x['smiles']} for x in c['candidates']]} for c in cases]


def compact_score(score, cache):
    keep=('row_id','rank','smiles','status','reason','energy_hartree','solvent',
          'computation_protocol_sha256','first_conformer_energy_hartree','conformer_spread_kcal_mol')
    result={k:score[k] for k in keep if k in score}
    fragments=[]
    for f in score.get('fragments',[]):
        configuration={k:f[k] for k in ('smiles','method','solvent','computation_protocol')}
        filename=xtb.cache_path(configuration,SimpleNamespace(cache=cache)).name
        fragments.append({'smiles':f['smiles'],'status':f['status'],'energy_hartree':f['energy_hartree'],
            'reason':f.get('reason'),'conformer_failures':f.get('failures',[]),
            'conformer_energies_hartree':f.get('conformer_energies_hartree'),
            'cache_record':('cache/'+filename) if (cache/filename).exists() else None})
    result['fragment_provenance']=fragments
    return result


def assess(cases, scores):
    by_row=defaultdict(list)
    for s in scores: by_row[s['row_id']].append(s)
    if set(by_row)!={c['source_row_id'] for c in cases}: raise ValueError('Incomplete row accounting')
    results={}
    for c in cases:
        items=sorted(by_row[c['source_row_id']],key=lambda s:s['rank'])
        if [(s['rank'],s['smiles']) for s in items]!=[(s['rank'],s['smiles']) for s in c['candidates']]:
            raise ValueError('Incomplete candidate accounting')
        result=pilot.energy_ranking(c,items,near_tie=1.)
        result['source_row_id']=c['source_row_id']; result['validation_line']=c['source_row_id']+1
        result['nominal_solvent']=c['solvent']['nominal_name']
        result['failure_details']=sorted({reason for s in items for f in s.get('fragment_provenance',[])
            for reason in ([f.get('reason')] + f.get('conformer_failures',[])) if reason})
        results[c['source_row_id']]=result
    complete=sum(r['status']=='fully_scored' for r in results.values())
    recover=sum(r['recovered'] for r in results.values())
    summary={'scope':'18 frozen unmatched validation failures only; not overall accuracy improvement',
        'selected_rows':len(cases),'complete_pool_rows':complete,'unresolved_rows':len(cases)-complete,
        'complete_pool_coverage':complete/len(cases), 'references_promoted_to_energy_rank1':recover,
        'recovery_all_selected':recover/len(cases),'recovery_fully_scored':recover/complete if complete else None,
        'near_tie_rows_among_complete':sum(len(r.get('near_tie_original_ranks',[]))>1 for r in results.values()),
        'conformer_winner_changes':sum(bool(r.get('conformer_sensitivity') and r['conformer_sensitivity']['winner_changes']) for r in results.values()),
        'failure_reasons_by_row':dict(Counter(reason for r in results.values() for reason in r['failure_details'])),
        'independent_chemical_assessment':'not performed', 'other_25_proposed_cases':'not run',
        'new_flowER_inference':False,'reranker_training':False,'cloud_spend':0,'reserved_test_outcomes_opened':False}
    return results,summary


def write_report(out,cases,results,summary):
    n=summary['selected_rows']; complete=summary['complete_pool_rows']; recovered=summary['references_promoted_to_energy_rank1']
    def pct(x):return f'{100*x:.1f}%' if x is not None else 'not defined (no complete pools)'
    lines=['# Energy-only focus-run results','',
        f"**{recovered}/{n} selected references were promoted to energy rank 1. Complete-pool coverage: {complete}/{n}.**",'',
        f"Recovery across all selected rows: **{pct(summary['recovery_all_selected'])}**. Recovery among fully scored rows: **{pct(summary['recovery_fully_scored'])}**. Unresolved rows retain the original frequency prediction and remain in the {n}-row denominator.",'',
        'This is the frozen unmatched-failure stratum only. It does not measure damage to correct baseline predictions or overall accuracy gain. Reference recovery is not independent chemical adjudication. The 25 competing-label cases have not been run.','',
        '[Complete candidate structures, energies and provenance](cases.html) · [Machine-readable case results](case-results.jsonl) · [Summary](summary.json) · [Frozen execution manifest](execution.json) · [Log](run.log)','',
        '| Validation line | Assumed solvent | Reference rank: frequency → energy | Recovered | Winner gap (kcal/mol) | Status / reason |',
        '| --- | --- | --- | --- | ---: | --- |']
    for c in cases:
        r=results[c['source_row_id']]; gap=r.get('winner_runner_up_gap_kcal_mol')
        lines.append(f"| {r['validation_line']} | {r['nominal_solvent']} | {r['reference_rank_before']} → {r['reference_rank_after'] if r['reference_rank_after'] is not None else 'unresolved'} | {'yes' if r['recovered'] else 'no'} | {f'{gap:.3f}' if gap is not None else '—'} | {r['status']}{': '+', '.join(r['failure_details']) if r['failure_details'] else ''} |")
    runtime=summary['runtime_and_cost']
    lines+=['',f"Elapsed physical-run time: **{runtime['elapsed_seconds']:.1f} seconds**. OS-reported CPU time (parent plus waited children): {runtime['cpu_seconds']:.1f} seconds. Two workers, one thread per worker. New cloud spend: **$0**.",'',
        f"Near-ties within the prespecified 1 kcal/mol band: {summary['near_tie_rows_among_complete']} complete rows. Winner changes using only the first prescribed conformer instead of the two-conformer minimum: {summary['conformer_winner_changes']}.",'',
        'Scores are isolated-fragment GFN2-xTB + ALPB total energies, including solvation once; they are not full Gibbs energies, barriers, rates or yields. Frozen nominal solvents are assumptions. Every retained candidate was accounted for; no failed case was replaced and no successfully scored subset of an incomplete row was ranked. No energy weight, switching margin or gap threshold was fitted.','',
        'The user authorized this bounded local focus run after the preparation freeze. Its execution receipt preserves that authorization separately; historical hold/proposal files were not altered. Training, new inference, cloud rentals and reserved-test access remain held. Failed fragment jobs were not retried.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,default=ROOT/'results/reranking/energy-only-frozen-solvents-20261031')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--xtb',type=Path,required=True)
    p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args();out=a.out
    authorization=json.loads(a.authorization.read_text())
    if not (authorization.get('local_physical_execution_authorized') is True and authorization.get('scope')==FOCUS and
            authorization.get('workers')==2 and authorization.get('maximum_elapsed_seconds')==7200):
        raise SystemExit('Explicit focus-run authorization receipt is required')
    if out.exists():raise SystemExit('Use a fresh run directory; no overwrite or automatic retry')
    cases,protocol,proposal_manifest=load_focus(a.proposal)
    executable=a.xtb.resolve()
    version=subprocess.run([str(executable),'--version'],capture_output=True,text=True,check=True,timeout=20)
    version_text=(version.stdout+'\n'+version.stderr).strip()
    binary_hash=pilot.sha(executable)
    jobs={(c['solvent']['solvent'],f) for c in cases for candidate in c['candidates'] for f in pilot.fragments(candidate['smiles'])}
    if len(jobs)!=111:raise ValueError('Expected the frozen focus plan of 111 fragment-solvent jobs')
    code={str(path.relative_to(ROOT)):pilot.sha(path) for path in [Path(__file__),ROOT/'scripts/score_xtb_candidates.py',ROOT/'scripts/energy_only_pilot.py',ROOT/'scripts/rerank_analysis.py']}
    out.mkdir(parents=True);cache=out/'cache';cache.mkdir()
    pilot.jsonl(out/'selected-cases.jsonl',cases)
    execution={'status':'frozen_before_physical_calculation','authorization':authorization,
        'authorization_sha256':pilot.sha(a.authorization),'proposal_manifest_sha256':pilot.sha(a.proposal/'manifest.json'),
        'parent_frozen_protocol':protocol,'scope':FOCUS,'row_ids':[c['source_row_id'] for c in cases],
        'selected_cases_sha256':pilot.sha(out/'selected-cases.jsonl'),'unique_fragment_solvent_jobs':len(jobs),
        'candidate_entries':sum(len(c['candidates']) for c in cases),'xtb_binary':str(executable),
        'xtb_version':version_text,'xtb_binary_sha256':binary_hash,'execution_code_sha256':code,
        'physical_hold_lifted_for_this_run_only':True,
        'remaining_holds':{k:v for k,v in protocol['execution_holds'].items() if k!='physical_calculations'}}
    pilot.save(out/'execution.json',execution)
    log=(out/'run.log').open('a',buffering=1)
    started=time.monotonic();deadline=started+7200
    def progress(message):
        elapsed=time.monotonic()-started
        print(f'[{elapsed:.1f}s] {message}',flush=True);log.write(f'[{elapsed:.1f}s] {message}\n')
        pilot.save(out/'progress.json',{'state':'running','elapsed_seconds':elapsed,'latest':message,
            'selected_rows':18,'completed_solvents':completed_solvents,'physical_deadline_seconds':7200})
    completed_solvents=[]; all_scores=[]; group_manifests=[]
    for solvent in sorted({c['solvent']['solvent'] for c in cases}):
        group=[c for c in cases if c['solvent']['solvent']==solvent]
        settings=SimpleNamespace(xtb=str(executable),xtb_version=version_text,xtb_sha256=binary_hash,
            solvent=solvent,reference_state='bar1mol',electronic_temperature=300.,conformers=2,
            timeout=20.,max_heavy_atoms=40,require_all_conformers=True,strict_elements=True,
            complete_pool_required=True,k=10,max_rows=0,jobs=2,max_unique_fragments=720,cache=cache,deadline=deadline)
        progress(f'Starting {solvent}: {len(group)} complete candidate pools')
        scores,info=xtb.score_records(physical_records(group),settings,progress=progress)
        pilot.jsonl(out/f'scores-{solvent}.jsonl',scores);pilot.save(out/f'manifest-{solvent}.json',info)
        all_scores += [compact_score(s,cache) for s in scores];group_manifests.append(info)
        completed_solvents.append(solvent)
        progress(f'Completed {solvent}: {info["candidate_status_counts"]}')
    results,summary=assess(cases,all_scores)
    elapsed=time.monotonic()-started
    own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    summary['runtime_and_cost']={'elapsed_seconds':elapsed,'cpu_seconds':own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,
        'workers':2,'threads_per_worker':1,'unique_fragment_solvent_jobs':111,
        'fragment_status_counts':dict(sum((Counter(info['fragment_status_counts']) for info in group_manifests),Counter())),
        'cloud_spend_usd':0,'electricity_cost':'not measured'}
    for name,h in code.items():
        if pilot.sha(ROOT/name)!=h:raise RuntimeError('Execution code changed during run')
    if pilot.sha(executable)!=binary_hash:raise RuntimeError('xTB binary changed during run')
    pilot.jsonl(out/'candidate-scores.jsonl',sorted(all_scores,key=lambda s:(s['row_id'],s['rank'])))
    pilot.jsonl(out/'case-results.jsonl',[results[c['source_row_id']] for c in cases]);pilot.save(out/'summary.json',summary)
    pilot.case_table(out,cases,results);write_report(out,cases,results,summary)
    progress(f'Finished: {summary["references_promoted_to_energy_rank1"]}/18 recovered; {summary["complete_pool_rows"]}/18 complete pools')
    pilot.save(out/'progress.json',{'state':'complete','summary':summary})
    hashes={p.name:pilot.sha(p) for p in out.iterdir() if p.is_file() and p.name!='run.log'}
    pilot.save(out/'completion.json',{'status':'complete','output_sha256':hashes,'source_proposal_unchanged':pilot.sha(a.proposal/'manifest.json')==execution['proposal_manifest_sha256']})
    log.close();print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
