#!/usr/bin/env python3
"""Validation-only group learning curve with frozen fold-specific policies."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

import learned_residual_model as model
import train_learned_residual as trainer
from phase0_group_power import bootstrap_ratios

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/reranking/learned-residual-validation-20261031'
FRACTIONS=(25,50,100)
SEEDS=tuple(20261031+i for i in range(5))
BOOTSTRAP_SEED=20261105
CPU_LIMIT=900.
VERSION='fixed_policy_group_learning_curve_v1'


def read_json(path):return json.loads(path.read_text())
def read_jsonl(path):return [json.loads(line) for line in path.read_text().splitlines()]
def snapshot(path):return {str(p.relative_to(path)):trainer.file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}


def verify_snapshot(base,expected):
    if snapshot(base)!=expected:raise ValueError('Original experiment changed')


def policy_from_selection(rows,outer,selection):
    expected=[r['source_row_id'] for r in rows if r['fold']!=outer]
    if selection['selection_source_row_ids']!=expected:
        raise ValueError('Policy was not selected exclusively within the outer fitting partition')
    return selection['selected']['config'].copy()


def fixed_policies(rows):
    # Never use the all-validation final policy to configure held-out folds.
    return {outer:policy_from_selection(rows,outer,read_json(BASE/f'outer{outer}-inner-selection.json'))
            for outer in range(5)}


def build_manifest(rows,policies,seeds=SEEDS):
    """Selection depends only on existing group/fold membership and fixed seeds."""
    jobs=[];comparisons=[]
    for outer,config in sorted(policies.items()):
        available=[i for i,r in enumerate(rows) if r['fold']!=outer]
        held=[i for i,r in enumerate(rows) if r['fold']==outer]
        if not available or not held:raise ValueError('Empty fitting or held-out partition')
        model.assert_disjoint(rows,available,held)
        groups=sorted({rows[i]['group'] for i in available})
        common=f'outer{outer}-pct100'
        jobs.append({'id':common,'outer_fold':outer,'fraction_percent':100,'subset_seed':None,
            'fit_ids':available,'fit_groups':groups,'available_groups':len(groups),'held_ids':held,'config':config})
        for seed in seeds:
            ordered=sorted(groups,key=lambda g:(hashlib.sha256(f'{seed}:{outer}:{g}'.encode()).hexdigest(),g))
            for fraction in (25,50):
                selected=sorted(ordered[:math.ceil(len(groups)*fraction/100)])
                group_set=set(selected);ids=[i for i in available if rows[i]['group'] in group_set]
                name=f'outer{outer}-pct{fraction}-seed{seed}'
                jobs.append({'id':name,'outer_fold':outer,'fraction_percent':fraction,'subset_seed':seed,
                    'fit_ids':ids,'fit_groups':selected,'available_groups':len(groups),'held_ids':held,'config':config})
                comparisons.append({'outer_fold':outer,'fraction_percent':fraction,'subset_seed':seed,'job':name})
            comparisons.append({'outer_fold':outer,'fraction_percent':100,'subset_seed':seed,'job':common})
    return {'jobs':jobs,'comparisons':comparisons,'seeds':list(seeds),'fractions':list(FRACTIONS)}


def verify_manifest(rows,manifest,policies):
    expected=build_manifest(rows,policies,tuple(manifest['seeds']))
    if expected!=manifest:raise ValueError('Group subset manifest changed')
    by_id={j['id']:j for j in manifest['jobs']}
    for outer in policies:
        for seed in manifest['seeds']:
            a=by_id[f'outer{outer}-pct25-seed{seed}'];b=by_id[f'outer{outer}-pct50-seed{seed}'];c=by_id[f'outer{outer}-pct100']
            if not set(a['fit_ids'])<=set(b['fit_ids'])<=set(c['fit_ids']):raise ValueError('Subsets are not nested')
            if a['held_ids']!=b['held_ids'] or b['held_ids']!=c['held_ids']:raise ValueError('Assessment populations differ')


def fresh_fit(rows,X,offsets,support,job,check_budget=lambda:None):
    model.assert_disjoint(rows,job['fit_ids'],job['held_ids'])
    check_budget()
    data=model.fit_inputs(rows,X,offsets,support,job['fit_ids'],job['config']['weighting'])
    fitted=model.fit(data,job['config']['l2'],check_budget)
    if not fitted['converged']:raise RuntimeError('Nonconvergent fit; no replacement subset: '+job['id'])
    return fitted


def aligned_outcomes(rows,outcomes):
    expected=[r['source_row_id'] for r in rows]
    ordered=sorted(outcomes,key=lambda r:r['source_row_id'])
    if [r['source_row_id'] for r in ordered]!=expected:raise ValueError('Missing or duplicated assessment rows')
    for row,result in zip(rows,ordered):
        if row['group']!=result['group'] or row['fold']!=result['fold']:raise ValueError('Assessment group mismatch')
    return ordered


def cluster_interval(groups,differences,seed=BOOTSTRAP_SEED,replicates=9999):
    """Average repeated fits per row, retaining the original number of clusters."""
    raw=np.asarray(differences)
    if raw.ndim!=2 or raw.shape[1]!=len(groups) or not raw.shape[0] or not np.isin(raw,[-1,0,1]).all():
        raise ValueError('Expected paired binary-outcome differences for the same rows')
    d=raw.astype(np.int64)
    _,inverse=np.unique(groups,return_inverse=True)
    sizes=np.bincount(inverse)
    totals=np.bincount(inverse,weights=d.sum(axis=0)).astype(np.int64)
    # Scaling numerator AND denominator by the repeat count permits the exact
    # integer compressed bootstrap. Cluster count stays len(sizes), not R*G.
    draws=bootstrap_ratios(sizes*d.shape[0],totals,np.random.default_rng(seed),replicates)
    return {'change':float(d.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
        'unique_assessment_rows':len(groups),'groups':len(sizes),'subset_repeats':d.shape[0],
        'bootstrap_replicates':replicates,'interval_scope':'held-group sampling conditional on the fitted models; repeats are not independent assessment rows'}


def prepare(out):
    if out.exists():raise FileExistsError('Fresh learning-curve directory required')
    started=time.process_time()
    parent=read_json(BASE/'protocol.json');trainer.check_sources(parent)
    prep=read_json(BASE/'preparation.json')
    for name,h in prep['artifact_sha256'].items():
        if trainer.file_hash(BASE/name)!=h:raise ValueError('Original prepared artifact changed: '+name)
    complete=read_json(BASE/'completion.json')
    if trainer.file_hash(BASE/'model/frozen.json')!=complete['model_receipt_sha256']:raise ValueError('Frozen model receipt changed')
    rows=read_jsonl(BASE/'validation-rows.jsonl')
    if [r['source_row_id'] for r in rows]!=list(range(21723)):raise ValueError('Validation population changed')
    policies=fixed_policies(rows);manifest=build_manifest(rows,policies)
    verify_manifest(rows,manifest,policies)
    protocol={'version':VERSION,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User requested training at 25%, 50%, 100% of fitting groups on the same separate groups',
        'base_experiment':str(BASE),'base_files_sha256':snapshot(BASE),'base_protocol_sha256':trainer.file_hash(BASE/'protocol.json'),
        'code_sha256':{str(p.relative_to(ROOT)):trainer.file_hash(p) for p in
            (Path(__file__),ROOT/'tests/test_learned_residual_learning_curve.py')},
        'runtime_versions':trainer.versions(),'policies':policies,
        'policy_provenance':'Existing inner-CV selection for each outer fitting partition only; no globally selected final policy, retuning or new model selection',
        'estimand':'Learning curve for coefficient/preprocessing/catalog fitting under fixed fold-specific policies, conditional on previous tuning using the full fitting partition',
        'limitation':'The smaller fractions constrain coefficient fitting data, not historical policy-tuning data; this is not an end-to-end label-budget curve',
        'fractions_percent':list(FRACTIONS),'subset_seeds':list(SEEDS),'outer_folds':5,
        'selection':'Label-blind SHA256(seed:outer:group) ordering; ceil(fraction*groups); complete nested groups; no outcome balancing',
        'fits':len(manifest['jobs']),'full_fraction':'One new 100% fit per outer fold, shared across subset-repeat comparisons; no old coefficients loaded',
        'feature_policy':'Reuse unchanged inference-only raw features; fit scaling, label catalog and weights within each selected fitting subset only',
        'assessment':'Identical held-out groups at every fraction/seed; every validation row retained once per curve repeat; original whole-pool fallback',
        'bootstrap':{'unit':'connected group','replicates':9999,'seed':BOOTSTRAP_SEED,
            'estimator':'row-weighted paired ratio of totals, averaged over five subset repeats without increasing the independent row/group count'},
        'primary_comparison':'100% minus 50%; 50% minus 25% and fraction-vs-frequency intervals are descriptive',
        'interpretation':'Report changes and intervals; failure to reject zero is not proof of a plateau; do not extrapolate beyond observed fitting sizes',
        'reference_worthwhile_gain':.01,'additional_diagnostics':'per-seed spread, training resubstitution accuracy, corrections/new errors/wrong-to-wrong/interventions',
        'nonconvergence':'Stop without replacing a subset or changing settings; report incomplete experiment',
        'cpu_limit_seconds':CPU_LIMIT,'compute':'Local sequential CPU, one numerical-library thread',
        'remaining_holds':parent['remaining_holds'],'reserved_test_outcomes_or_candidates_opened':False}
    out.mkdir(parents=True);trainer.save(out/'protocol.json',protocol);trainer.save(out/'selection-manifest.json',manifest)
    trainer.save(out/'preparation.json',{'protocol_sha256':trainer.file_hash(out/'protocol.json'),
        'manifest_sha256':trainer.file_hash(out/'selection-manifest.json'),'preparation_cpu_seconds':time.process_time()-started,
        'jobs':len(manifest['jobs']),'held_out_rows_per_curve_repeat':len(rows)})
    print(json.dumps({'status':'prepared','fits':len(manifest['jobs']),'held_rows_per_repeat':len(rows),
        'policies':policies,'groups_per_full_fit':{j['outer_fold']:len(j['fit_groups']) for j in manifest['jobs'] if j['fraction_percent']==100}},indent=2),flush=True)


def validate_preparation(out):
    p=read_json(out/'protocol.json');prep=read_json(out/'preparation.json')
    if trainer.file_hash(out/'protocol.json')!=prep['protocol_sha256'] or trainer.file_hash(out/'selection-manifest.json')!=prep['manifest_sha256']:
        raise ValueError('Frozen learning-curve preparation changed')
    for name,h in p['code_sha256'].items():
        if trainer.file_hash(ROOT/name)!=h:raise ValueError('Learning-curve code changed')
    if trainer.versions()!=p['runtime_versions']:raise ValueError('Runtime changed')
    trainer.check_sources(read_json(BASE/'protocol.json'));verify_snapshot(BASE,p['base_files_sha256'])
    return p,prep


def run(out,resume=False):
    if (out/'completion.json').exists():raise FileExistsError('Completed experiment cannot be overwritten')
    protocol,prep=validate_preparation(out)
    previous=prep['preparation_cpu_seconds']
    if (out/'progress.json').exists():
        if not resume:raise ValueError('Use --resume for existing checkpoints')
        previous=read_json(out/'progress.json')['cpu_seconds_used']
    started=time.process_time();wall=time.monotonic()
    rows=read_jsonl(BASE/'validation-rows.jsonl');X=sparse.load_npz(BASE/'features.npz')
    offsets=np.load(BASE/'offsets.npy',allow_pickle=False)
    support=[r['supported'] for r in read_json(BASE/'feature-support.json')]
    manifest=read_json(out/'selection-manifest.json');policies=fixed_policies(rows)
    verify_manifest(rows,manifest,policies)
    for name in ('fits','held-out'): (out/name).mkdir(exist_ok=True)
    log=(out/'run.log').open('a',buffering=1)
    def used():return previous+time.process_time()-started
    def budget():
        if used()>CPU_LIMIT:raise trainer.BudgetExpired('Learning-curve 900-CPU-second cap reached')
    def progress(message):
        print(message,flush=True);log.write(f'[{used():.1f} CPU seconds] {message}\n')
        trainer.save(out/'progress.json',{'status':'running','cpu_seconds_used':used(),'latest':message})
    try:
        with threadpool_limits(limits=1):
            for number,job in enumerate(manifest['jobs'],1):
                budget();name=job['id'];path=out/'fits'/name
                binding=model.digest({'protocol_sha256':prep['protocol_sha256'],'manifest_sha256':prep['manifest_sha256'],'job':job})
                receipt_path,_=trainer.checkpoint_paths(path)
                if receipt_path.exists():fitted=trainer.load_checkpoint(path,binding)
                else:
                    fitted=fresh_fit(rows,X,offsets,support,job,budget)
                    trainer.save_checkpoint(path,binding,fitted,[rows[i]['source_row_id'] for i in job['fit_ids']])
                if not fitted['converged']:raise RuntimeError('Nonconvergent checkpoint')
                assessments={}
                for split,ids in [('held',job['held_ids']),('fitting',job['fit_ids'])]:
                    values=model.corrections(fitted,X,offsets,ids)
                    picks=model.choices(rows,ids,values,support,job['config'])
                    results=model.outcomes(rows,ids,picks,support)
                    assessments[split]=model.summarize(results)
                    if split=='held':
                        if job['fraction_percent']==100:
                            original=read_jsonl(BASE/f'outer{job["outer_fold"]}-held-out.jsonl')
                            if results!=original:raise ValueError('Fresh 100% copy does not reproduce original held-out predictions')
                        trainer.jsonl(out/'held-out'/f'{name}.jsonl',results)
                trainer.save(out/'fits'/f'{name}-assessment.json',{'job':name,'fit_rows':len(job['fit_ids']),
                    'fit_groups':len(job['fit_groups']),'held_groups':len({rows[i]['group'] for i in job['held_ids']}),
                    'configuration':job['config'],'summaries':assessments,'training_summary_is_resubstitution':True})
                progress(f'Fit {number}/{len(manifest["jobs"])} {name}: {len(job["fit_ids"])} fitting rows, {len(job["fit_groups"])} groups, converged')
        budget();summary=summarize_curve(rows,manifest,out)
        validate_preparation(out);budget()
        summary['runtime']={'cpu_seconds':used(),'run_elapsed_seconds':time.monotonic()-wall,'cloud_cost_usd':0,'gpu_used':False}
        trainer.save(out/'summary.json',summary);write_report(out,summary,protocol)
        receipt={'status':'complete','all_fits_converged':len(manifest['jobs']),'all_fits_initialized_from_zero':True,
            'fresh_100_percent_predictions_reproduce_original':True,'source_artifacts_unchanged':True,
            'unique_assessment_rows_per_fraction_repeat':len(rows),'reserved_test_outcomes_or_candidates_opened':False,
            'summary_sha256':trainer.file_hash(out/'summary.json'),'protocol_sha256':prep['protocol_sha256'],
            'cpu_seconds_used':used(),'primary_frozen_model_replaced':False}
        trainer.save(out/'completion.json',receipt)
        progress('Learning curve complete; original model preserved; no new inference or cloud execution')
        trainer.save(out/'progress.json',dict(receipt,latest='complete'))
    except Exception as exc:
        progress(type(exc).__name__+': '+str(exc))
        trainer.save(out/'incomplete.json',{'error':type(exc).__name__,'reason':str(exc),'cpu_seconds_used':used(),
            'no_subsets_replaced':True,'no_protocol_changes':True})
        raise
    finally:log.close()


def summarize_curve(rows,manifest,out):
    jobs={j['id']:j for j in manifest['jobs']};seeds=manifest['seeds']
    groups=[r['group'] for r in rows]
    baseline=np.array([bool(r['candidates']) and r['candidates'][0]['smiles']==r['target'] for r in rows],dtype=np.int64)
    matrices={};points=[]
    for fraction in FRACTIONS:
        repeat_summaries=[];correct=[]
        for seed in seeds:
            refs=[c for c in manifest['comparisons'] if c['fraction_percent']==fraction and c['subset_seed']==seed]
            outcomes=[]
            for ref in refs:outcomes.extend(read_jsonl(out/'held-out'/(ref['job']+'.jsonl')))
            ordered=aligned_outcomes(rows,outcomes)
            correct.append([int(r['correct']) for r in ordered])
            repeat_summaries.append(dict(model.summarize(ordered),subset_seed=seed))
        matrices[fraction]=np.asarray(correct,dtype=np.int64)
        paired=cluster_interval(groups,matrices[fraction]-baseline[None,:])
        selected=[j for j in jobs.values() if j['fraction_percent']==fraction]
        assessments=[read_json(out/'fits'/(j['id']+'-assessment.json')) for j in selected]
        total_training_rows=sum(a['summaries']['fitting']['rows'] for a in assessments)
        point={'fraction_percent':fraction,'mean_accuracy':float(matrices[fraction].mean()),
            'baseline_accuracy':float(baseline.mean()),'paired_vs_frequency':paired,'per_seed':repeat_summaries,
            'seed_accuracy_range':[min(s['accuracy'] for s in repeat_summaries),max(s['accuracy'] for s in repeat_summaries)],
            'mean_counts':{key:float(np.mean([s[key] for s in repeat_summaries])) for key in
                ('correct','correction','new_error','wrong_to_wrong','changed','fallback_unsupported','empty_pool')},
            'fitting_groups':{'mean':float(np.mean([len(j['fit_groups']) for j in selected])),
                'range':[min(len(j['fit_groups']) for j in selected),max(len(j['fit_groups']) for j in selected)]},
            'fitting_rows':{'mean':float(np.mean([len(j['fit_ids']) for j in selected])),
                'range':[min(len(j['fit_ids']) for j in selected),max(len(j['fit_ids']) for j in selected)]},
            'training_resubstitution_accuracy':sum(a['summaries']['fitting']['correct'] for a in assessments)/total_training_rows,
            'training_loss_rows':{'mean':float(np.mean([read_json(out/'fits'/(j['id']+'.json'))['info']['loss_rows'] for j in selected]))}}
        points.append(point)
    comparisons=[]
    for small,large in ((25,50),(50,100)):
        result=cluster_interval(groups,matrices[large]-matrices[small])
        result.update(smaller_percent=small,larger_percent=large,primary=large==100,
            per_seed_change=[float(np.mean(a-b)) for a,b in zip(matrices[large],matrices[small])])
        comparisons.append(result)
    return {'points':points,'comparisons':comparisons,'unique_assessment_rows':len(rows),'groups':len(set(groups)),
        'fits':len(jobs),'subset_seeds':seeds,'effective_assessment_rows_not_multiplied_by_repeats':True}


def write_report(out,s,protocol):
    lines=['# Validation learning curve under fixed fold policies','',
        'Each point assesses the same 21,723 official-validation rows in their existing five outer held-out folds. Smaller fractions use five label-blind, nested samples of complete fitting groups. All 55 models start from zero; the five fresh 100% models reproduce the original held-out decisions exactly.','',
        '| Fitting groups | Mean fitting groups / rows per model | Held-out accuracy | Change vs frequency (95% paired CI), pp | Corrections / new errors, mean |',
        '| --- | ---: | ---: | ---: | ---: |']
    for p in s['points']:
        lo,hi=p['paired_vs_frequency']['ci95'];counts=p['mean_counts']
        lines.append(f'| {p["fraction_percent"]}% | {p["fitting_groups"]["mean"]:.1f} / {p["fitting_rows"]["mean"]:.1f} | {100*p["mean_accuracy"]:.3f}% | {100*p["paired_vs_frequency"]["change"]:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] | {counts["correction"]:.1f} / {counts["new_error"]:.1f} |')
    lines+=['',f'Frequency accuracy is {100*s["points"][0]["baseline_accuracy"]:.3f}% at every point. A fraction is a fraction of each outer fitting partition, not a fraction of the held-out data. All assessment rows, baseline-correct cases, unsupported cases and empty pools are retained. Mean counts at smaller fractions are averages over five repeats, not counts from extra independent evaluation rows.','',
        '## Paired increments','', '| Increase in fitting groups | Accuracy change, pp | 95% paired group interval, pp |', '| --- | ---: | ---: |']
    for c in s['comparisons']:
        lo,hi=c['ci95'];lines.append(f'| {c["smaller_percent"]}% → {c["larger_percent"]}% | {100*c["change"]:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |')
    lines+=['','The primary increment is 50% → 100%; other intervals are descriptive. Confidence intervals use 9,999 paired connected-group bootstrap draws with ratio-of-row-totals weighting. Repeated subset fits are averaged within each original assessment row; they do not multiply the 21,723 unique assessment rows or 2,873 groups. These intervals condition on the fitted models and do not include a complete refitting bootstrap. Per-seed spreads and changes are retained in [summary.json](summary.json).','',
        '## Interpretation and limitations','',
        'This is a learning curve for fitting coefficients, scales and soft-positive catalogs under fixed fold-specific settings. Those settings came from the previous inner CV using each full fitting partition. They do not use the outer held-out labels, but the 25% and 50% points still benefit from that historical tuning. This is therefore not a curve for a strictly capped total label budget. No policy is retuned or selected from this curve.','',
        'Prior validation and exploratory diagnostics informed the features; this remains development evidence. The reserved 20,000 test outcomes are unopened. A flat or uncertain last increment is not proof that additional data cannot help. Increasing performance would support benefit over the measured range, without predicting performance beyond 100%. Judge the last paired interval and subset variation before deciding whether to obtain more data.','',
        '## Training and preservation','',
        'The input-only feature matrix, generator, candidate strings/counts/order and all earlier artifacts are unchanged. Within each selected fitting subset, scaling, label catalogs, soft labels and bounded/uniform weights are recomputed. Multiple recorded outcomes remain positive. Unsupported whole pools retain frequency, and rows without fitting positives are omitted only from the loss. Original fold-specific regularization, weighting, lambda and margin remain fixed.','',
        '| Outer held-out fold | Fixed configuration |','| --- | --- |']
    for fold,c in protocol['policies'].items():lines.append(f'| {fold} | `{json.dumps(c,sort_keys=True)}` |')
    lines+=['',f'Local CPU used: {s["runtime"]["cpu_seconds"]:.1f} seconds. Run elapsed time: {s["runtime"]["run_elapsed_seconds"]:.1f} seconds. No GPU or cloud spending. The prior frozen model is not replaced. Energy, DeepMech, new FlowER inference, cloud execution and reserved test outcomes remain held.','',
        '[Frozen protocol](protocol.json) · [Selection manifest](selection-manifest.json) · [Results](summary.json) · [Run log](run.log) · [Completion checks](completion.json)','',
        '## Commands','', 'Run tests from the repository root:','', '```bash',
        '.venv-analysis/bin/python -m unittest discover -s tests -p test_learned_residual_learning_curve.py','```','',
        'Executed commands (preparation requires a fresh directory; a completed run cannot be overwritten):','', '```bash',
        f'.venv-analysis/bin/python scripts/learned_residual_learning_curve.py prepare --out {out.relative_to(ROOT) if out.is_absolute() else out}',
        f'.venv-analysis/bin/python scripts/learned_residual_learning_curve.py run --out {out.relative_to(ROOT) if out.is_absolute() else out}','```','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action=='prepare':prepare(args.out)
        else:run(args.out,args.resume)
