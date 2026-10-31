#!/usr/bin/env python3
"""Execute a frozen v2 preparation only with new manifest-bound authorization."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import resource
import shutil
import time

import energy_only_pilot as pilot
import prepare_energy_focus_v2 as preparation
import run_energy_focus as focus
import score_xtb_candidates as old
import score_xtb_candidates_v2 as adapter


def authorize(receipt, manifest_sha):
    if not (receipt.get('local_physical_execution_authorized') is True and
            receipt.get('manifest_sha256')==manifest_sha and receipt.get('workers')==2 and
            receipt.get('maximum_elapsed_seconds')==7200):
        raise ValueError('New explicit authorization bound to the frozen v2 manifest is required')


def aggregate(cases,results,cache):
    scores=[]
    for case in cases:
        solvent=case['solvent']['solvent']
        for c in case['candidates']:
            if old.inventory(c['smiles'])!=old.inventory(case['reactant']):
                raise ValueError('Frozen candidate inventory changed')
            fragments=[results[preparation.job_id(solvent,f)] for f in pilot.fragments(c['smiles'])]
            valid=all(f['status']=='ok' and old.valid_result(f) for f in fragments)
            cfg=adapter.protocol(preparation.settings(solvent,cache))
            score={'row_id':case['source_row_id'],'rank':c['rank'],'smiles':c['smiles'],
                'solvent':solvent,'status':'ok' if valid else 'failed','energy_hartree':None,
                'computation_protocol_sha256':pilot.fingerprint(cfg),'fragments':fragments}
            if valid:
                values=[f['conformer_energies_hartree'] for f in fragments]
                score.update(energy_hartree=sum(f['energy_hartree'] for f in fragments),
                    first_conformer_energy_hartree=sum(v[0] for v in values),
                    conformer_spread_kcal_mol=sum(max(v)-min(v) for v in values)*pilot.HARTREE_KCAL)
            else:score['reason']='one_or_more_fragments_unscored'
            scores.append(focus.compact_score(score,cache))
    return scores


def execute(prepared,out,authorization):
    receipt=json.loads(authorization.read_text())
    authorize(receipt,pilot.sha(prepared/'manifest.json'))
    manifest,cases=preparation.verify_freeze(prepared)
    if out.exists():raise ValueError('Fresh run directory required; no automatic retries')
    plan=[json.loads(l) for l in (prepared/'fragment-plan.jsonl').read_text().splitlines()]
    if len(plan)!=111 or len({p['job_id'] for p in plan})!=111:raise ValueError('Unexpected fragment plan')
    out.mkdir(parents=True);cache=out/'cache';cache.mkdir()
    shutil.copyfile(prepared/'selected-cases.jsonl',out/'selected-cases.jsonl')
    pilot.save(out/'execution.json',{'prepared_manifest_sha256':pilot.sha(prepared/'manifest.json'),
        'authorization':receipt,'authorization_sha256':pilot.sha(authorization),
        'physical_hold_lifted_for_this_frozen_run_only':True,'selected_rows':18,'candidates':62,
        'ranking':'energy only; full-pool fallback; no label-dependent tuning'})
    results={};pending=[]
    for job in plan:
        if job['action']=='reuse_validated_success':
            source=prepared/job['imported_cache'];data=json.loads(source.read_text())
            if data['status']!='ok' or not old.valid_result(data):raise ValueError('Invalid imported success')
            shutil.copyfile(source,cache/source.name);results[job['job_id']]=data
        else:pending.append(job)
    started=time.monotonic();deadline=started+7200
    with (out/'run.log').open('a',buffering=1) as log:
        def progress(message):
            line=f'[{time.monotonic()-started:.1f}s] {message}'
            print(line,flush=True);log.write(line+'\n')
        progress(f'{len(results)} validated jobs reused; {len(pending)} new jobs authorized')
        iterator=iter(pending);exhausted=False
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures={}
            while futures or not exhausted:
                while len(futures)<2 and not exhausted:
                    try:job=next(iterator)
                    except StopIteration:exhausted=True;break
                    args=preparation.settings(job['solvent'],cache);args.deadline=deadline
                    futures[executor.submit(adapter.fragment_energy,job['smiles'],args)]=(job,args)
                if not futures:continue
                done,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    job,args=futures.pop(future)
                    try:data=future.result()
                    except Exception as exc:
                        data=old.failure(adapter.configuration(job['smiles'],args),'worker_exception',exc)
                        old.atomic_json(old.cache_path(adapter.configuration(job['smiles'],args),args),data)
                    results[job['job_id']]=data
                    progress(f'Processed {len(results)}/111 jobs; current status {data["status"]}')
        scores=aggregate(cases,results,cache);assessed,base=focus.assess(cases,scores)
        elapsed=time.monotonic()-started
        own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        summary={'version':adapter.VERSION,'calculation_coverage':{
            'fragment_jobs':111,'reused_validated_jobs':111-len(pending),
            'new_job_status_counts':dict(Counter(results[p['job_id']]['status'] for p in pending)),
            'fully_scored_candidates':sum(s['status']=='ok' for s in scores),'candidate_entries':62,
            'complete_pools':base['complete_pool_rows'],'selected_pools':18,'unresolved_pools':base['unresolved_rows']},
            'reference_recovery':{k:base[k] for k in ('references_promoted_to_energy_rank1','recovery_all_selected','recovery_fully_scored')},
            'near_tie_rows':base['near_tie_rows_among_complete'],'conformer_winner_changes':base['conformer_winner_changes'],
            'failure_reasons_by_row':base['failure_reasons_by_row'],
            'runtime_and_cost':{'elapsed_seconds':elapsed,'cpu_seconds':own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,
                'workers':2,'cloud_cost_usd':0},'overall_accuracy_claim':False}
        preparation.verify_freeze(prepared)
        pilot.jsonl(out/'candidate-scores.jsonl',scores)
        pilot.jsonl(out/'case-results.jsonl',[assessed[c['source_row_id']] for c in cases])
        pilot.save(out/'summary.json',summary);pilot.case_table(out,cases,assessed)
        (out/'REPORT.md').write_text('# V2 energy-only results\n\nCalculation coverage and reference recovery are separate.\n\n'
            f"Complete calculation pools: **{base['complete_pool_rows']}/18**. Fully scored candidates: **{sum(s['status']=='ok' for s in scores)}/62**.\n\n"
            f"Reference recovery: **{base['references_promoted_to_energy_rank1']}/18 selected cases**, and **{base['references_promoted_to_energy_rank1']}/{base['complete_pool_rows']} fully scored cases** (undefined when no pools complete).\n\n"
            'Unresolved rows retain the frequency winner. Original results, candidates and frozen solvents are unchanged. '
            'No failed original calculation energy was imported, and no new failure was automatically retried. '
            'This failure-only pilot cannot estimate overall accuracy improvement.\n\n'
            '[Candidate structures, energies and provenance](cases.html) · [Summary](summary.json) · [Per-case results](case-results.jsonl)\n')
        progress('Completed; coverage and reference recovery saved separately')
    pilot.save(out/'completion.json',{'status':'complete','output_sha256':preparation.tree_hashes(out),
        'original_results_unchanged':True,'prepared_manifest_sha256':pilot.sha(prepared/'manifest.json')})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','out','authorization'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();execute(a.prepared,a.out,a.authorization)
