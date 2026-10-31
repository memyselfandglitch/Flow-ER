#!/usr/bin/env python3
"""Fresh grouped development experiment on validation + inspected test prefix.

Only saved, previously inspected outcomes are eligible. Reserved test inputs
containing targets are never opened. Earlier models and protocols stay intact.
"""
import os
for _thread_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_var] = '1'

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

import learned_residual_features as features
import learned_residual_model as model
import train_learned_residual as prior
from audit_recorded_targets import input_functions
from phase0_reranking_audit import model_input_hash, digest, save
from phase0_group_power import bootstrap_ratios

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'results/reranking/learned-residual-validation-20261031'
PREFIX = ROOT / 'results/reranking/failure-study/records.jsonl'
BASE = ROOT / 'results/reranking/full-validation-test20k-20261029'
META = ROOT / 'results/reranking/full-validation-new-model-20261029'
PROVENANCE = ROOT / 'results/reranking/phased-20261031/provenance.json'
VERSION = 'combined_validation_inspected_prefix_residual_v1'
SEED = 20261031
CPU_LIMIT = 7200.
ALLOWED = {PREFIX, PROVENANCE, META/'protocol.json', META/'test-evaluation-rows.jsonl',
           BASE/'validation-rows.json', BASE/'test-rows.json', BASE/'protocol.json'}
ALLOWED |= {PREVIOUS / name for name in ('protocol.json', 'preparation.json',
    'validation-rows.jsonl', 'features.npz', 'offsets.npy', 'feature-support.json')}


def read_allowed(path):
    path = Path(path).resolve()
    if path not in ALLOWED:
        raise PermissionError('Not a combined-development input: ' + str(path))
    return path.read_bytes()


def read_json(path):
    return json.loads(read_allowed(path))


def code_hashes():
    paths = [Path(__file__), ROOT/'tests/test_combined_residual.py',
             ROOT/'scripts/audit_recorded_targets.py', ROOT/'utils/data_utils.py']
    return prior.code_hashes() | {str(p.relative_to(ROOT)): digest(p) for p in paths}


def checked_pool(record):
    if record['sample_count'] != 64:
        raise ValueError('Expected original 64-draw pools')
    cs = [{k:c[k] for k in ('rank', 'smiles', 'count')} for c in record['candidates'][:10]]
    counts = [c['count'] for c in cs]
    if (counts != sorted(counts, reverse=True) or sum(counts)>64 or
            any(type(n) is not int or n<=0 for n in counts) or
            len({c['smiles'] for c in cs}) != len(cs) or
            [c['rank'] for c in cs] != list(range(1,len(cs)+1))):
        raise ValueError('Candidate identity/order/count mismatch')
    if 'target_rank' in record:
        ranks = [c['rank'] for c in cs if c['smiles']==record['target']]
        rank = record['target_rank']
        if ranks != ([rank] if rank is not None and rank<=10 else []):
            raise ValueError('Target rank differs from preserved pool')
    return cs


def group_rows(rows):
    """Union across sources before assigning folds; outcomes are not consulted."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i,r in enumerate(rows):
        keys = [('base',r['base_group']), ('exact',r['exact_input']),
                ('normalized',r['normalized_input'])]
        if str(r['raw_sequence_id']).isdigit():
            keys.append(('network',r['source_split'],str(r['raw_sequence_id'])))
        for key in keys:
            if key in seen:
                a,b=find(i),find(seen[key]); parent[max(a,b)]=min(a,b)
            seen[key]=i
    members=defaultdict(list)
    for i in range(len(rows)):members[find(i)].append(i)
    for ids in members.values():
        first=min(rows[i]['row_uid'] for i in ids)
        key=hashlib.sha256(f'{VERSION}:{SEED}:{first}'.encode()).hexdigest()
        for i in ids:rows[i].update(group=key,fold=int(key[:16],16)%5)
    if len({r['row_uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cross-source row identity')
    for fold in range(5):
        fitting=[i for i,r in enumerate(rows) if r['fold']!=fold]
        held=[i for i,r in enumerate(rows) if r['fold']==fold]
        if not fitting or not held:raise ValueError('Empty outer fold')
        model.assert_disjoint(rows,fitting,held)
    return Counter(len(ids) for ids in members.values())


def reserved_guard(rows,reserved):
    overlap = {
        'rows':len({(r['source_split'],r['source_row_id']) for r in rows} &
                   {(r['source_split'],r['source_row_id']) for r in reserved}),
        'normalized_inputs':len({r['normalized_input'] for r in rows} &
                                 {r['reactant_key'] for r in reserved}),
        'base_components':len({r['base_group'] for r in rows} &
                               {r['split_group'] for r in reserved})}
    if any(overlap.values()):raise ValueError('Reserved test overlap: '+str(overlap))
    return {'overlap':overlap,'reserved_rows':len(reserved),
            'scope':'input hashes and global input/network component metadata only',
            'reserved_exact_condition_arrays_recomputed':False,
            'reserved_targets_or_candidates_opened':False}


def load_rows(progress):
    previous_protocol=read_json(PREVIOUS/'protocol.json')
    prior.check_sources(previous_protocol)
    preparation=read_json(PREVIOUS/'preparation.json')
    for name,h in preparation['artifact_sha256'].items():
        if digest(PREVIOUS/name)!=h:raise ValueError('Earlier preparation changed: '+name)
    provenance=read_json(PROVENANCE)['inputs']
    for path in (PREFIX, BASE/'validation-rows.json',BASE/'test-rows.json'):
        if digest(path)!=provenance[str(path.relative_to(ROOT))]:
            raise ValueError('Previously frozen source changed: '+str(path))
    meta_protocol=read_json(META/'protocol.json')
    if digest(META/'test-evaluation-rows.jsonl')!=meta_protocol['manifest_sha256']['test-evaluation-rows.jsonl']:
        raise ValueError('Test input-only metadata changed')
    valmeta={r['source_row_id']:r for r in read_json(BASE/'validation-rows.json')}
    # This metadata contains no targets/candidate outcomes, including beyond the prefix.
    metabytes=read_allowed(META/'test-evaluation-rows.jsonl')
    testmeta={r['source_row_id']:r for r in map(json.loads,metabytes.splitlines())
              if r['source_row_id']<20152}
    validation=[json.loads(l) for l in read_allowed(PREVIOUS/'validation-rows.jsonl').splitlines()]
    if [r['source_row_id'] for r in validation]!=list(range(21723)):
        raise ValueError('Incomplete validation preparation')
    rows=[]
    for r in validation:
        m=valmeta[r['source_row_id']]
        rows.append(dict(r,source_split='val',cohort='validation',row_uid=f"val:{r['source_row_id']}",
            source_line=r['source_row_id']+1,base_group=m['split_group'],raw_sequence_id=m['raw_sequence_id']))
    tokenize,matrix=input_functions(ROOT/'utils/data_utils.py'); cache={}
    prefix=[json.loads(l) for l in read_allowed(PREFIX).splitlines()]
    if [r['source_row_id'] for r in prefix]!=list(range(20152)):
        raise ValueError('Only the previously inspected 20,152-row prefix is authorized')
    for i,r in enumerate(prefix):
        m=testmeta[r['source_row_id']]
        if r['source_split']!='test' or r['source_line_sha256']!=m['source_line_sha256']:
            raise ValueError('Prefix source binding differs')
        normalized=hashlib.sha256(r['reactant'].encode()).hexdigest()
        if normalized!=m['reactant_key'] or normalized!=r['reactant_key']:
            raise ValueError('Prefix input identity differs')
        source=r['reaction'].split('>>',1)[0]
        if source not in cache:cache[source]=model_input_hash(source,tokenize,matrix)
        rows.append({'source_split':'test','cohort':'inspected_test_prefix',
            'source_row_id':r['source_row_id'],'row_uid':f"test:{r['source_row_id']}",
            'source_line':r['source_row_id']+1,'reactant':r['reactant'],'target':r['target'],
            'sample_count':64,'candidates':checked_pool(r),'normalized_input':normalized,
            'exact_input':cache[source],'base_group':m['split_group'],
            'raw_sequence_id':m['raw_sequence_id'],'original_pool_sha256':model.digest(r['candidates']),
            'source_file':str(PREFIX.relative_to(ROOT))})
        if (i+1)%5000==0:progress(f'Prefix input identities: {i+1}/20152')
    for r in rows:checked_pool(r)
    groups=group_rows(rows)
    guard=reserved_guard(rows,read_json(BASE/'test-rows.json'))
    return rows,groups,guard


def check_sources(protocol):
    if code_hashes()!=protocol['code_sha256'] or prior.versions()!=protocol['runtime_versions']:
        raise ValueError('Implementation/runtime changed after preparation')
    for name,h in protocol['source_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Prepared source changed: '+name)
    for name,h in protocol['preserved_artifact_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Earlier frozen artifact changed: '+name)


def prepare(out):
    if out.exists():raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True);start=time.process_time()
    with (out/'run.log').open('a',buffering=1) as log:
        def progress(message):print(message,flush=True);log.write(message+'\n')
        rows,groups,guard=load_rows(progress)
        protected=list((PREVIOUS/'model').iterdir())+[PREVIOUS/'REPORT.md',PREVIOUS/'completion.json']
        sources={str(p.relative_to(ROOT)):digest(p) for p in ALLOWED}
        sources.update(read_json(PREVIOUS/'protocol.json')['source_sha256'])
        protocol={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
            'authorization':'User requested training/CV on prior test prefix and selected combined 41,875 development rows',
            'source_sha256':sources,'code_sha256':code_hashes(),'runtime_versions':prior.versions(),
            'preserved_artifact_sha256':{str(p.relative_to(ROOT)):digest(p) for p in protected},
            'rows':len(rows),'source_counts':dict(Counter(r['cohort'] for r in rows)),
            'seed':SEED,'outer_folds':5,'inner_folds':4,'grid':model.grid(),
            'grouping':'transitive global input/network components, exact model-condition arrays, normalized inputs; source-local numeric network IDs',
            'fold_assignment':'SHA256(version:seed:smallest group row UID) modulo 5; input-only',
            'score':'log(count/64) + lambda * linear residual; unchanged prior feature schema',
            'labels':'fitting-fold empirical soft positives; no assessment labels in fitting catalog',
            'primary':'combined row-weighted paired strict top-1 difference; 9999 complete-group bootstrap draws',
            'secondary':'per-source differences/intervals descriptive; no model selection from outer results',
            'final_refit':'select via combined development grouped CV, then fit all development rows from zero',
            'interpretation':'Previously inspected ordered test prefix is repurposed as development; all results exploratory, not final test accuracy',
            'cpu_seconds_limit':CPU_LIMIT,'wall_seconds_limit':7200,'library_threads':1,
            'remaining_holds':['new FlowER inference','cloud execution','reserved-test outcomes','physical calculations','DeepMech'],
            'reserved_guard':guard,'feature_names':features.FEATURE_NAMES}
        save(out/'protocol.json',protocol);prior.jsonl(out/'development-rows.jsonl',rows)
        # Pure inference features can be reused; no scaler, coefficients or labels are reused.
        oldX=sparse.load_npz(PREVIOUS/'features.npz')
        oldoffsets=np.load(PREVIOUS/'offsets.npy',allow_pickle=False)
        oldinfo=read_json(PREVIOUS/'feature-support.json')
        Xp,op,ip=features.build(rows[21723:],lambda s:progress(s.replace('validation rows','prefix rows')))
        X=sparse.vstack([oldX,Xp],format='csr');offsets=np.r_[oldoffsets,op[1:]+oldoffsets[-1]]
        infos=oldinfo+ip
        if X.shape[0]!=sum(len(r['candidates']) for r in rows) or len(offsets)!=len(rows)+1:
            raise ValueError('Feature/candidate alignment differs')
        sparse.save_npz(out/'features.npz',X);np.save(out/'offsets.npy',offsets,allow_pickle=False)
        save(out/'feature-support.json',infos)
        check_sources(protocol)
        receipt={'rows':len(rows),'candidates':X.shape[0],'features':X.shape[1],
            'groups':sum(groups.values()),'group_size_histogram':dict(groups),
            'fold_rows':dict(Counter(r['fold'] for r in rows)),
            'feature_support':dict(Counter(i['reason'] for i in infos)),
            'feature_reuse':'21,723 validation inference-feature rows only; all fits/scalers/catalogs fresh',
            'reserved_guard':guard,'preparation_cpu_seconds':time.process_time()-start,
            'artifact_sha256':{n:digest(out/n) for n in ('protocol.json','development-rows.jsonl',
                'features.npz','offsets.npy','feature-support.json')}}
        save(out/'preparation.json',receipt);progress(json.dumps(receipt,sort_keys=True))


def enriched_outcomes(rows,indices,picks,support):
    result=model.outcomes(rows,indices,picks,support)
    for item,i in zip(result,indices):
        item.update({k:rows[i][k] for k in ('row_uid','source_split','source_line','cohort')})
    return result


def summarize(outcomes):
    summary=model.summarize(outcomes);totals=defaultdict(lambda:[0,0])
    for r in outcomes:totals[r['group']][0]+=1;totals[r['group']][1]+=r['difference']
    draws=bootstrap_ratios([v[0] for v in totals.values()],[v[1] for v in totals.values()],
                           np.random.default_rng(SEED),9999)
    summary.update(groups=len(totals),paired_group_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist())
    return summary


def train(out):
    p=json.loads((out/'protocol.json').read_text());prep=json.loads((out/'preparation.json').read_text())
    check_sources(p)
    for name,h in prep['artifact_sha256'].items():
        if digest(out/name)!=h:raise ValueError('Prepared artifact changed: '+name)
    if (out/'fits').exists() or (out/'model').exists():raise FileExistsError('Training already started; preserve this run')
    rows=[json.loads(l) for l in (out/'development-rows.jsonl').read_text().splitlines()]
    X=sparse.load_npz(out/'features.npz');offsets=np.load(out/'offsets.npy',allow_pickle=False)
    support=[i['supported'] for i in json.loads((out/'feature-support.json').read_text())]
    (out/'fits').mkdir();start=time.process_time();wall=time.monotonic();fit_count=0
    log=(out/'run.log').open('a',buffering=1)
    def budget():
        if prep['preparation_cpu_seconds']+time.process_time()-start>CPU_LIMIT or time.monotonic()-wall>7200:
            raise prior.BudgetExpired('Combined experiment compute cap reached')
    def progress(message):
        print(message,flush=True);log.write(message+'\n')
        save(out/'progress.json',{'status':'running','latest':message,'fits':fit_count,
             'cpu_seconds':prep['preparation_cpu_seconds']+time.process_time()-start,
             'training_wall_seconds':time.monotonic()-wall})
    def fitting(ids,l2,weight,name,data=None):
        nonlocal fit_count
        budget();data=data if data is not None else model.fit_inputs(rows,X,offsets,support,ids,weight)
        fitted=model.fit(data,l2,budget)
        binding=model.digest({'rows':[(rows[i]['row_uid'],rows[i]['target'],rows[i]['normalized_input']) for i in ids],
            'l2':l2,'weighting':weight,'features':prep['artifact_sha256']['features.npz'],'code':p['code_sha256']})
        prior.save_checkpoint(out/'fits'/name,binding,fitted,[rows[i]['row_uid'] for i in ids]);fit_count+=1
        progress(f'Fit {name}: converged={fitted["converged"]}, iterations={fitted["info"]["iterations"]}')
        return fitted
    def cv(ids,folds,nfold,name):
        def callback(fit_ids,l2,weight,fold,data):
            return fitting(fit_ids,l2,weight,f'{name}-fold{fold}-{weight}-l2_{l2}',data)
        chosen,grid,rejected=model.cross_validated_policy(rows,X,offsets,support,ids,folds,nfold,callback)
        save(out/(name+'-selection.json'),{'selected':chosen,'grid':grid,'rejected':rejected,
            'selection_row_uids':[rows[i]['row_uid'] for i in ids]})
        return chosen
    try:
        outer=[]
        with threadpool_limits(limits=1):
            for fold in range(5):
                fitting_ids=[i for i,r in enumerate(rows) if r['fold']!=fold]
                held=[i for i,r in enumerate(rows) if r['fold']==fold]
                model.assert_disjoint(rows,fitting_ids,held)
                chosen=cv(fitting_ids,model.inner_folds(rows,fitting_ids,SEED+fold),4,f'outer{fold}-inner')
                config=chosen['config'];fitted=fitting(fitting_ids,config['l2'],config['weighting'],f'outer{fold}-refit')
                if not fitted['converged']:raise RuntimeError('Outer refit did not converge')
                scores=model.corrections(fitted,X,offsets,held)
                result=enriched_outcomes(rows,held,model.choices(rows,held,scores,support,config),support)
                outer.extend(result);prior.jsonl(out/f'outer{fold}-held-out.jsonl',result)
                save(out/f'outer{fold}-result.json',{'config':config,'summary':model.summarize(result),
                    'held_out_row_uids':[rows[i]['row_uid'] for i in held]})
                progress(f'Outer fold {fold+1}/5: {json.dumps(model.summarize(result))}')
            ids=list(range(len(rows)))
            chosen=cv(ids,{i:r['fold'] for i,r in enumerate(rows)},5,'final-development-cv')
            config=chosen['config'];final=fitting(ids,config['l2'],config['weighting'],'final-all-development')
            if not final['converged']:raise RuntimeError('Final fit did not converge')
        by_uid={r['row_uid']:r for r in outer}
        if len(outer)!=len(rows) or set(by_uid)!={r['row_uid'] for r in rows}:
            raise ValueError('Missing/duplicated held-out decisions')
        outer=[by_uid[r['row_uid']] for r in rows]
        summary={'combined':summarize(outer),'by_cohort':{cohort:summarize([r for r in outer if r['cohort']==cohort])
            for cohort in ('validation','inspected_test_prefix')},
            'interpretation':p['interpretation'],'secondary_intervals':'descriptive, unadjusted'}
        prior.jsonl(out/'nested-oof-decisions.jsonl',outer);save(out/'development-summary.json',summary)
        labels=model.catalog(rows,ids);diagnostic=defaultdict(list)
        for r,o in zip(rows,outer):
            diagnostic['multiple_recorded_targets' if len(labels[r['normalized_input']])>1 else 'single_recorded_target'].append(o)
        save(out/'ambiguity-diagnostic.json',{'catalog':'all combined development outcomes; post-hoc diagnosis only',
            'strata':{k:model.summarize(v) for k,v in diagnostic.items()}})
        check_sources(p);budget();destination=out/'model';destination.mkdir()
        np.savez_compressed(destination/'weights.npz',weights=final['weights'],scale=final['scale'])
        save(destination/'label-catalog.json',labels)
        freeze={'version':VERSION,'configuration':config,'training_rows':len(rows),'training_info':final['info'],
            'model_sha256':digest(destination/'weights.npz'),'label_catalog_sha256':digest(destination/'label-catalog.json'),
            'protocol_sha256':digest(out/'protocol.json'),'feature_names':features.FEATURE_NAMES,
            'code_sha256':p['code_sha256'],'runtime_versions':prior.versions(),'frozen_utc':datetime.now(timezone.utc).isoformat(),
            'fits':fit_count,'cpu_seconds':prep['preparation_cpu_seconds']+time.process_time()-start,
            'training_wall_seconds':time.monotonic()-wall,'reserved_targets_or_candidates_opened':False,
            'generator_changed':False,'zero_intervention_selected':config['lambda']==0,
            'remaining_holds':p['remaining_holds']}
        save(destination/'frozen.json',freeze)
        save(destination/'coefficients.json',[{'feature':name,'standardized_coefficient':float(w)}
            for name,w in zip(features.FEATURE_NAMES,final['weights'])])
        write_report(out,p,prep,summary,freeze)
        for path in destination.iterdir():path.chmod(0o444)
        save(out/'completion.json',{'status':'complete','model_receipt_sha256':digest(destination/'frozen.json'),
            'summary_sha256':digest(out/'development-summary.json'),'sources_and_previous_model_preserved':True})
        progress('Combined experiment complete; new model frozen; reserved outcomes remain sealed')
        save(out/'progress.json',{'status':'complete','fits':fit_count,'cpu_seconds':freeze['cpu_seconds']})
    finally:log.close()


def write_report(out,protocol,prep,summary,freeze):
    lines=['# Combined development reranker','',
        'Fresh linear frequency-residual model on official validation and the previously inspected ordered test prefix. '
        'These are exploratory development results, not an independent test benchmark.','',
        '| Cohort | Rows | Frequency | Reranked | Gain (pp) | 95% group interval (pp) | Corrections | New errors |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for name,s in [('Combined (primary)',summary['combined'])]+list(summary['by_cohort'].items()):
        lo,hi=s['paired_group_bootstrap_95ci']
        lines.append(f'| {name} | {s["rows"]:,} | {100*s["baseline_accuracy"]:.3f}% | {100*s["accuracy"]:.3f}% | '
            f'{100*s["paired_change"]:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {s["correction"]} | {s["new_error"]} |')
    lines += ['',f'Connected development groups: {prep["groups"]:,}. Every row receives exactly one outer-held-out decision. '
        'Five outer folds contain four inner folds for policy selection. Labels, scalers and coefficients use fitting rows only. '
        'The 9,999-draw paired bootstrap resamples complete connected groups and uses row totals. Source-specific intervals are descriptive.',
        '',f'Final policy: `{json.dumps(freeze["configuration"],sort_keys=True)}`. '
        f'Fresh fits: {freeze["fits"]}; total preparation/training CPU seconds: {freeze["cpu_seconds"]:.1f}. Local CPU; $0 cloud.',
        '', 'The same existing feature schema and tuning grid were retained. Validation inference features were reused unchanged; '
        'no earlier weights, scalers or fitting-label catalogs were reused. Saved top-10 candidates and counts were preserved. '
        'Unavailable features/empty pools remain in the denominator with baseline fallback.',
        '', 'The original frozen validation model is preserved. Reserved 20,000-row test membership, normalized-input and global '
        'network-component overlap checks pass using metadata only. Reserved exact model-condition arrays were not recomputed. '
        'No reserved labels or candidates, new FlowER inference, physical calculations, DeepMech or cloud execution were used.',
        '', 'Earlier analysis of both sources influenced feature design. The prefix is ordered and not representative of the full test set. '
        'This run does not isolate the causal effect of adding the prefix because folds and the assessed population differ from the prior run. '
        'The final all-development policy selection score is not an independent estimate and is not substituted for nested held-out results.',
        '', '[Protocol](protocol.json) · [Summary](development-summary.json) · [Decisions](nested-oof-decisions.jsonl) · '
        '[Frozen model](model/frozen.json) · [Post-hoc ambiguity diagnostic](ambiguity-diagnostic.json)', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else train)(args.out.resolve())


if __name__=='__main__':main()
