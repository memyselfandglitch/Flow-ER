#!/usr/bin/env python3
"""Fresh validation-only residual reranker; reserved outcomes/inference stay held."""
import os
for _thread_var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_var]='1'

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy import sparse
from rdkit import rdBase
from threadpoolctl import threadpool_limits

import learned_residual_features as features
import learned_residual_model as model
from phase0_group_power import bootstrap_ratios
from phase0_reranking_audit import digest as file_hash,save
from rerank_analysis import canonical

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/reranking/full-validation-test20k-20261029'
AUDIT=ROOT/'results/reranking/phased-20261031'
VAL=ROOT/'data/flower_new_dataset/val.txt'
SOURCES=[BASE/'cache/val.jsonl']+[BASE/f'inference/records/val{i:03}.jsonl' for i in range(5)]
VERSION='validation_soft_positive_frequency_residual_v1'
SEED=20261031
CPU_LIMIT=7200.
ALLOWED=set(SOURCES+[VAL,AUDIT/'provenance.json',AUDIT/'validation-identities.json',
    AUDIT/'reproduction-verification.json',BASE/'validation-rows.json'])


def read_allowed(path):
    path=Path(path).resolve()
    if path not in ALLOWED:raise PermissionError('Not a validation-training input: '+str(path))
    return path.read_bytes()


def versions():return {'numpy':np.__version__,'scipy':scipy.__version__,'rdkit':rdBase.rdkitVersion}


def code_hashes():
    names=['train_learned_residual.py','learned_residual_model.py','learned_residual_features.py',
        'phased_reranking_contracts.py','targeted_reranker.py','rerank_analysis.py','phase0_group_power.py','phase0_reranking_audit.py']
    paths=[ROOT/'scripts'/n for n in names]+[ROOT/'tests/test_learned_residual.py']
    return {str(path.relative_to(ROOT)):file_hash(path) for path in paths}


def checkpoint_paths(path):return Path(str(path)+'.json'),Path(str(path)+'.npz')


def load_checkpoint(path,binding):
    receipt_path,weights_path=checkpoint_paths(path)
    receipt=json.loads(receipt_path.read_text())
    if receipt['binding']!=binding or file_hash(weights_path)!=receipt['weights_sha256']:
        raise ValueError('Fit checkpoint mismatch')
    with np.load(weights_path,allow_pickle=False) as stored:
        return {'weights':stored['weights'],'scale':stored['scale'],'converged':receipt['info']['converged'],'info':receipt['info']}


def save_checkpoint(path,binding,fitted,source_ids):
    receipt_path,weights_path=checkpoint_paths(path)
    np.savez_compressed(weights_path,weights=fitted['weights'],scale=fitted['scale'])
    save(receipt_path,{'binding':binding,'weights_sha256':file_hash(weights_path),'info':fitted['info'],
        'fitting_source_row_ids':source_ids})


def jsonl(path,values):
    with path.open('w') as stream:
        for value in values:stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def load_validation():
    provenance=json.loads(read_allowed(AUDIT/'provenance.json'))
    identity_raw=read_allowed(AUDIT/'validation-identities.json')
    receipt=json.loads(read_allowed(AUDIT/'reproduction-verification.json'))
    if hashlib.sha256(identity_raw).hexdigest()!=receipt['files']['validation-identities.json']['sha256']:
        raise ValueError('Frozen Phase 0 grouping changed')
    identities={r['source_row_id']:r for r in json.loads(identity_raw)}
    lines=read_allowed(VAL).splitlines(keepends=True)
    if len(lines)!=21723:raise ValueError('Official validation size changed')
    rows=[];seen=set();sources={};linked={};canon={}
    metadata_path=BASE/'validation-rows.json';metadata_raw=read_allowed(metadata_path)
    if hashlib.sha256(metadata_raw).hexdigest()!=provenance['inputs'][str(metadata_path.relative_to(ROOT))]:
        raise ValueError('Validation metadata changed')
    metadata={r['source_row_id']:r for r in json.loads(metadata_raw)}
    def normalized(s):
        if s not in canon:canon[s]=canonical(s)
        return canon[s]
    for path in SOURCES:
        data=read_allowed(path);h=hashlib.sha256(data).hexdigest();relative=str(path.relative_to(ROOT))
        if h!=provenance['inputs'][relative]:raise ValueError('Validation source changed: '+relative)
        sources[relative]=h
        for raw in data.splitlines():
            r=json.loads(raw);i=r['source_row_id'];ident=identities[i];meta=metadata[i]
            if i in seen or r['source_split']!='val' or r['sample_count']!=64:raise ValueError('Invalid row identity/sampling')
            seen.add(i)
            if hashlib.sha256(lines[i]).hexdigest()!=r['source_line_sha256']:raise ValueError('Source line changed')
            source,target=lines[i].decode().strip().split('|')[0].split('>>')
            if normalized(source)!=r['reactant'] or normalized(target)!=r['target']:raise ValueError('Validation input/target mismatch')
            if hashlib.sha256(r['reactant'].encode()).hexdigest()!=ident['normalized_input']:raise ValueError('Normalized identity mismatch')
            if r['split_group']!=meta['split_group']:raise ValueError('Network metadata mismatch')
            cs=[{k:c[k] for k in ('rank','smiles','count')} for c in r['candidates']]
            if len(cs)>10 or len({c['smiles'] for c in cs})!=len(cs):raise ValueError('Candidate pool changed')
            counts=[c['count'] for c in cs]
            if counts!=sorted(counts,reverse=True) or any(type(n)!=int or n<=0 for n in counts) or sum(counts)>64:
                raise ValueError('Candidate counts changed')
            if [c['rank'] for c in cs]!=list(range(1,len(cs)+1)):raise ValueError('Candidate ranking changed')
            ranks=[c['rank'] for c in cs if c['smiles']==r['target']]
            if ranks!=([r['target_rank']] if r['target_rank'] is not None and r['target_rank']<=10 else []):
                raise ValueError('Saved target rank differs')
            row={'source_row_id':i,'reactant':r['reactant'],'target':r['target'],'sample_count':64,'candidates':cs,
                'normalized_input':ident['normalized_input'],'exact_input':ident['exact_input'],
                'group':ident['phase0_group'],'fold':ident['fold'],'original_pool_sha256':model.digest(r['candidates']),
                'source_file':relative,'source_record_sha256':hashlib.sha256(raw).hexdigest()}
            keys=[('normalized',ident['normalized_input']),('exact',ident['exact_input']),('network_component',meta['split_group'])]
            if str(r.get('raw_sequence_id','')).isdigit():keys.append(('network',str(r['raw_sequence_id'])))
            for key in keys:
                if key in linked and linked[key]!=(row['group'],row['fold']):raise ValueError('Connected group crosses folds')
                linked[key]=(row['group'],row['fold'])
            rows.append(row)
    if seen!=set(range(21723)) or sum(len(r['candidates']) for r in rows)!=49600:raise ValueError('Incomplete validation candidates')
    for path in (VAL,AUDIT/'validation-identities.json',BASE/'validation-rows.json'):
        sources[str(path.relative_to(ROOT))]=file_hash(path)
    return sorted(rows,key=lambda r:r['source_row_id']),sources


def check_sources(protocol):
    for name,h in protocol['source_sha256'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('Frozen data changed: '+name)
    if code_hashes()!=protocol['code_sha256'] or versions()!=protocol['runtime_versions']:
        raise ValueError('Frozen implementation/runtime changed')


def prepare(out):
    if out.exists():raise FileExistsError('Fresh experiment directory required')
    out.mkdir(parents=True)
    started=time.process_time();rows,sources=load_validation()
    protocol={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User: OKAY lets proceed with the learned reranker',
        'scope':'new nonphysical linear model, grouped validation development and all-validation refit only',
        'training_rows':21723,'candidate_entries':49600,'source_sha256':sources,
        'code_sha256':code_hashes(),'runtime_versions':versions(),'feature_names':features.FEATURE_NAMES,
        'score':'log(candidate_count/64) + lambda * linear_correction; no intercept',
        'labels':'fitting-fold normalized-input catalog; soft positives over all recorded candidates; missing positive skips loss only',
        'features':'map-free full-mixture structural descriptors, local fixed-hash fingerprints, inference-available frequency; no energy/rarity/reference mappings',
        'outer_folds':5,'inner_folds':4,'folds':'frozen Phase 0 connected exact/normalized/network groups',
        'seed':SEED,'inner_assignment':'SHA256(seed + outer_fold + group), modulo 4; no labels used',
        'grid':model.grid(),'initialization':'zero coefficients for every independent fit; no previous model loaded',
        'optimizer':{'name':'L-BFGS-B','maxiter':500,'ftol':1e-11,'gtol':1e-6,'maxls':40,'reject_nonconvergence':True},
        'scaler':'fitting-candidate standard deviations only, no mean subtraction; constant scales=1',
        'minimum_worthwhile_gain':.01,'bootstrap':{'replicates':9999,'seed':SEED,'unit':'connected group','estimator':'ratio of paired totals'},
        'cpu_seconds_limit':CPU_LIMIT,'compute':'local sequential CPU; one numerical-library thread',
        'remaining_holds':['DeepMech','energy calculations','new FlowER inference','cloud execution','reserved-test outcomes'],
        'test_plan':'fixed existing 20,000 steps; candidate generation pending, targets unopened',
        'final_refit':'validation-only selection followed by all-validation fit; freeze even if selected lambda=0',
        'interpretation':'nested OOF is development evidence; prior validation inspection precludes calling it pristine test performance'}
    save(out/'protocol.json',protocol);jsonl(out/'validation-rows.jsonl',rows)
    with (out/'run.log').open('a',buffering=1) as log:
        def progress(message):print(message,flush=True);log.write(message+'\n')
        X,offsets,infos=features.build(rows,progress)
        sparse.save_npz(out/'features.npz',X);np.save(out/'offsets.npy',offsets,allow_pickle=False)
        save(out/'feature-support.json',infos)
    check_sources(protocol)
    receipt={'rows':len(rows),'candidates':X.shape[0],'features':X.shape[1],'nonzero_features':X.nnz,
        'groups':len({r['group'] for r in rows}),'fold_rows':dict(Counter(r['fold'] for r in rows)),
        'feature_support':dict(Counter(r['reason'] for r in infos)),
        'artifact_sha256':{n:file_hash(out/n) for n in ('protocol.json','validation-rows.jsonl','features.npz','offsets.npy','feature-support.json')},
        'preparation_cpu_seconds':time.process_time()-started,'reserved_test_targets_or_candidates_opened':False}
    save(out/'preparation.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


class BudgetExpired(RuntimeError):pass


def train(out,resume=False):
    protocol=json.loads((out/'protocol.json').read_text());preparation=json.loads((out/'preparation.json').read_text())
    check_sources(protocol)
    for name,h in preparation['artifact_sha256'].items():
        if file_hash(out/name)!=h:raise ValueError('Preparation artifact changed')
    if (out/'model/frozen.json').exists():raise FileExistsError('Already frozen; no refitting after freeze')
    if (out/'progress.json').exists() and not resume:raise ValueError('Use --resume for existing fit checkpoints')
    previous=json.loads((out/'progress.json').read_text())['cpu_seconds_used'] if (out/'progress.json').exists() else preparation['preparation_cpu_seconds']
    started=time.process_time();wall=time.monotonic();rows=[json.loads(l) for l in (out/'validation-rows.jsonl').read_text().splitlines()]
    X=sparse.load_npz(out/'features.npz');offsets=np.load(out/'offsets.npy',allow_pickle=False)
    infos=json.loads((out/'feature-support.json').read_text());support=[r['supported'] for r in infos]
    (out/'fits').mkdir(exist_ok=True)
    log=(out/'run.log').open('a',buffering=1)
    def budget():
        if previous+time.process_time()-started>CPU_LIMIT:raise BudgetExpired('Authorized two-CPU-hour limit reached')
    def progress(message):
        used=previous+time.process_time()-started
        print(message,flush=True);log.write(f'[{used:.1f} CPU seconds] {message}\n')
        save(out/'progress.json',{'status':'running','cpu_seconds_used':used,'latest':message})
    def fitting(indices,l2,weight,name,data=None):
        budget();path=out/'fits'/name;receipt_path,weights_path=checkpoint_paths(path)
        binding=model.digest({'fitting_rows':[(rows[i]['source_row_id'],rows[i]['target'],rows[i]['normalized_input']) for i in indices],
            'features_sha256':preparation['artifact_sha256']['features.npz'],'l2':l2,'weighting':weight,'code':protocol['code_sha256']})
        if receipt_path.exists():
            return load_checkpoint(path,binding)
        data=data or model.fit_inputs(rows,X,offsets,support,indices,weight)
        fitted=model.fit(data,l2,budget)
        save_checkpoint(path,binding,fitted,[rows[i]['source_row_id'] for i in indices])
        progress(f'Fit {name}: converged={fitted["converged"]}, iterations={fitted["info"]["iterations"]}')
        return fitted
    def cv(indices,folds,nfold,prefix):
        def callback(ids,l2,weight,fold,data):
            return fitting(ids,l2,weight,f'{prefix}-fold{fold}-{weight}-l2_{l2}',data)
        selected,reports,rejected=model.cross_validated_policy(rows,X,offsets,support,indices,folds,nfold,callback)
        save(out/(prefix+'-selection.json'),{'selected':selected,'grid':reports,
            'rejected_unconverged_configurations':sorted(rejected),'selection_source_row_ids':[rows[i]['source_row_id'] for i in indices]})
        return selected
    try:
        outer_results=[]
        with threadpool_limits(limits=1):
            for outer in range(5):
                fit_ids=[i for i,r in enumerate(rows) if r['fold']!=outer];held=[i for i,r in enumerate(rows) if r['fold']==outer]
                model.assert_disjoint(rows,fit_ids,held)
                selected=cv(fit_ids,model.inner_folds(rows,fit_ids,SEED+outer),4,f'outer{outer}-inner')
                c=selected['config'];fitted=fitting(fit_ids,c['l2'],c['weighting'],f'outer{outer}-refit')
                if not fitted['converged']:raise RuntimeError('Selected outer model did not converge')
                values=model.corrections(fitted,X,offsets,held)
                decisions=model.choices(rows,held,values,support,c)
                outcomes=model.outcomes(rows,held,decisions,support);outer_results.extend(outcomes)
                jsonl(out/f'outer{outer}-held-out.jsonl',outcomes)
                save(out/f'outer{outer}-result.json',{'config':c,'summary':model.summarize(outcomes),
                    'held_out_labels_used_for_training_or_selection':False})
                progress(f'Outer fold {outer+1}/5 complete: {model.summarize(outcomes)}')
            all_ids=list(range(len(rows)));selected=cv(all_ids,{i:r['fold'] for i,r in enumerate(rows)},5,'final-validation-cv')
            c=selected['config'];final=fitting(all_ids,c['l2'],c['weighting'],'final-all-validation')
            if not final['converged']:raise RuntimeError('All-validation refit failed convergence')
        outer_results.sort(key=lambda r:r['source_row_id'])
        if [r['source_row_id'] for r in outer_results]!=list(range(21723)):raise ValueError('OOF rows missing/duplicated')
        summary=model.summarize(outer_results);grouped=defaultdict(lambda:[0,0])
        for r in outer_results:grouped[r['group']][0]+=1;grouped[r['group']][1]+=r['difference']
        vals=bootstrap_ratios([v[0] for v in grouped.values()],[v[1] for v in grouped.values()],np.random.default_rng(SEED),9999)
        interval=np.quantile(vals,[.025,.975]).tolist()
        summary.update(paired_group_bootstrap_95ci=interval,groups=len(grouped),bootstrap_replicates=9999,
            interpretation='Nested grouped validation development estimate; prior validation inspection disclosed')
        jsonl(out/'nested-oof-decisions.jsonl',outer_results);save(out/'development-summary.json',summary)
        # Catalog agreement is explicitly outcome-side diagnosis after predictions.
        labels=model.catalog(rows,all_ids);sets=sum(r['prediction'] in labels[rows[i]['normalized_input']] for i,r in enumerate(outer_results))
        save(out/'set-agreement-diagnostic.json',{'agreement_rows':sets,'rows':len(rows),
            'catalog':'all official-validation recorded targets, including each assessed row; diagnostic only; not used in outer fitting or selection'})
        budget();check_sources(protocol)
        model_dir=out/'model';model_dir.mkdir()
        np.savez_compressed(model_dir/'weights.npz',weights=final['weights'],scale=final['scale'])
        save(model_dir/'label-catalog.json',labels)
        freeze={'version':VERSION,'configuration':c,'feature_names':features.FEATURE_NAMES,
            'model_sha256':file_hash(model_dir/'weights.npz'),'label_catalog_sha256':file_hash(model_dir/'label-catalog.json'),
            'protocol_sha256':file_hash(out/'protocol.json'),'code_sha256':code_hashes(),'runtime_versions':versions(),
            'validation_source_sha256':protocol['source_sha256'],'training_rows':21723,'training_info':final['info'],
            'frozen_utc':datetime.now(timezone.utc).isoformat(),'reference_test_outcomes_opened':False,
            'generator_changed':False,'candidate_pools_changed':False,'zero_intervention_selected':c['lambda']==0,
            'fallback':'unsupported whole-row features or empty pool retain baseline; strict switching margin; stable original ties',
            'budget':{'cpu_seconds_used':previous+time.process_time()-started,'training_process_elapsed_seconds':time.monotonic()-wall},
            'remaining_holds':protocol['remaining_holds'],'final_test_candidates':'not yet generated; new inference remains held'}
        save(model_dir/'frozen.json',freeze)
        weights=[{'feature':name,'standardized_coefficient':float(w),'raw_coefficient':float(w/scale)} for name,w,scale in zip(features.FEATURE_NAMES,final['weights'],final['scale'])]
        save(model_dir/'coefficients.json',sorted(weights,key=lambda r:-abs(r['standardized_coefficient'])))
        write_report(out,summary,freeze,preparation)
        for path in model_dir.iterdir():path.chmod(0o444)
        progress('Training complete; fresh all-validation model frozen; reserved test remains untouched')
        save(out/'progress.json',{'status':'complete','cpu_seconds_used':freeze['budget']['cpu_seconds_used'],
            'model':str(model_dir/'weights.npz'),'development':summary})
        save(out/'completion.json',{'status':'complete','model_receipt_sha256':file_hash(model_dir/'frozen.json'),
            'development_summary_sha256':file_hash(out/'development-summary.json'),'source_and_code_verified_unchanged':True})
    except BudgetExpired as exc:
        progress(str(exc));save(out/'budget-exhausted.json',{'reason':str(exc),'resume_requires_new_budget_authorization':True});raise
    finally:log.close()


def write_report(out,s,f,p):
    lo,hi=s['paired_group_bootstrap_95ci'];c=f['configuration']
    text=f'''# Fresh learned residual reranker

All 21,723 official-validation rows and 49,600 unchanged saved candidate entries were accounted for. A new linear frequency-residual model was fitted with fitting-fold soft-positive catalogs, uniform versus bounded-failure weights, and nested connected-group cross-validation. No previous model weights, energy scores, rarity rewards or reference-derived mappings were used.

## Nested grouped development results

- Frequency accuracy: {s['baseline_correct']}/{s['rows']} ({100*s['baseline_accuracy']:.3f}%).
- Reranked accuracy: {s['correct']}/{s['rows']} ({100*s['accuracy']:.3f}%).
- Paired change: {100*s['paired_change']:+.3f} percentage points; 95% group-bootstrap interval [{100*lo:+.3f}, {100*hi:+.3f}] points.
- Corrections: {s['correction']}; new errors: {s['new_error']}; wrong-to-wrong switches: {s['wrong_to_wrong']}.
- Interventions: {s['changed']}/{s['rows']}; unsupported-feature fallback: {s['fallback_unsupported']}; empty pools: {s['empty_pool']}.
- Groups: {s['groups']}; bootstrap replicates: 9,999, ratio of row totals.

These are development results. Prior validation diagnostics informed feature design; this is not an untouched test result. Each outer policy was selected using only its fitting partition's inner folds. The final policy below was selected separately using all-validation grouped CV, then refitted on all validation; its selection score is not substituted for the nested estimate.

## Frozen all-validation model

Configuration: `{json.dumps(c,sort_keys=True)}`. Zero intervention selected: **{f['zero_intervention_selected']}**.

Model: [weights.npz](model/weights.npz), with separate coefficients and fitting-only scale arrays; [freeze receipt](model/frozen.json), [coefficient table](model/coefficients.json), and [training label catalog](model/label-catalog.json). Every independent fit starts from zero. Positive labels for competing recorded outcomes are retained; absence from a catalog is not chemical invalidity.

Features cover the complete represented mixture using map-free counts, functional groups, hydrogen/charge/bond-count differences, fixed hashed local fingerprints, frequency/gaps/diversity and interpretable structural warnings. Molecular presence does not establish experimental solvent conditions. Unsupported whole pools retain the original frequency prediction. No mapped atom edits or inferred proton-transfer mechanism is claimed.

CPU consumed: {f['budget']['cpu_seconds_used']:.1f} seconds; training-process elapsed time: {f['budget']['training_process_elapsed_seconds']:.1f} seconds. Local CPU only; cloud cost $0. Feature support: `{json.dumps(p['feature_support'],sort_keys=True)}`.

## Reserved test evaluation remains pending

The fixed 20,000-step test membership is preserved. Its candidate generation remains pending and new FlowER inference/cloud execution remain held. No reserved target was opened. This model is frozen before any future test evaluation; do not retune it on test results.

[Development summary](development-summary.json) · [Nested out-of-fold decisions](nested-oof-decisions.jsonl) · [Final validation-only selection](final-validation-cv-selection.json) · [Protocol](protocol.json) · [Log](run.log)
'''
    (out/'REPORT.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','train'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':
        with threadpool_limits(limits=1):prepare(a.out)
    else:train(a.out,a.resume)
