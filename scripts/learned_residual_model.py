"""Grouped, soft-positive linear frequency residual. No legacy weights used."""
from collections import Counter,defaultdict
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from phased_reranking_contracts import soft_targets,bounded_failure_weight,residual_choice

L2_GRID=(.01,.1,1.,10.)
LAMBDAS=(.1,.25,.5,1.)
MARGINS=(0.,.1,.25,.5)
WEIGHTINGS=('uniform','bounded_failure')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def catalog(rows,indices):
    labels=defaultdict(Counter)
    for i in indices:labels[rows[i]['normalized_input']][rows[i]['target']]+=1
    return {key:dict(sorted(counts.items())) for key,counts in sorted(labels.items())}


def candidate_indices(offsets,indices):
    return np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in indices]) if len(indices) else np.array([],dtype=int)


def fit_inputs(rows,X,offsets,support,indices,weighting):
    labels=catalog(rows,indices);flat=[];ys=[];priors=[];lengths=[];weights=[];covered=[];skipped=Counter()
    for i in indices:
        r=rows[i];cs=r['candidates']
        if not support[i]:skipped['unsupported_features']+=1;continue
        if not cs:skipped['empty_pool']+=1;continue
        q,mass=soft_targets([c['smiles'] for c in cs],labels[r['normalized_input']]);covered.append(mass)
        if q is None:skipped['no_fitting_catalog_positive_in_pool']+=1;continue
        flat.extend(range(offsets[i],offsets[i+1]));lengths.append(len(cs));ys.extend(q)
        priors.extend(np.log([c['count']/64. for c in cs]))
        weights.append(bounded_failure_weight(q,weighting=='bounded_failure'))
    # Unsupervised scaling sees fitting candidates only, including fitting rows
    # with no recoverable reference. All unsupported rows fall back as a whole.
    scale_ids=candidate_indices(offsets,[i for i in indices if support[i]])
    Z=X[scale_ids]
    mean=np.asarray(Z.mean(axis=0)).ravel() if len(scale_ids) else np.zeros(X.shape[1])
    variance=np.asarray(Z.power(2).mean(axis=0)).ravel()-mean**2 if len(scale_ids) else np.ones(X.shape[1])
    scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-8]=1.
    lengths=np.asarray(lengths,dtype=int)
    if not len(lengths):raise ValueError('No eligible fitting rows')
    starts=np.r_[0,np.cumsum(lengths)[:-1]];owner=np.repeat(np.arange(len(lengths)),lengths)
    weights=np.asarray(weights);emphasis_count=int(np.count_nonzero(weights==2));weights=weights/weights.mean()
    fitX=X[np.asarray(flat,dtype=int)].multiply(1/scale).tocsr()
    info={'fitting_rows':len(indices),'loss_rows':len(lengths),'loss_candidates':len(flat),
        'scaling_candidates':len(scale_ids),'skipped_loss_rows':dict(skipped),'catalog_inputs':len(labels),
        'catalog_sha256':digest(labels),'mean_fitting_catalog_mass_covered':float(np.mean(covered)) if covered else None,
        'mean_fitting_catalog_mass_omitted':float(np.mean(1-np.asarray(covered))) if covered else None,
        'raw_emphasis_rows':emphasis_count if weighting=='bounded_failure' else 0,
        'weighting':weighting,'no_positive_policy':'skip loss only; retain in evaluation',
        'catalog_provenance':'fitting indices only; every recorded outcome remains positive'}
    return {'X':fitX,'q':np.asarray(ys),'prior':np.asarray(priors),'starts':starts,'owner':owner,
        'weights':weights,'scale':scale,'catalog':labels,'info':info}


def loss_gradient(w,data,l2):
    score=data['prior']+data['X']@w
    maxima=np.maximum.reduceat(score,data['starts'])
    exp=np.exp(score-maxima[data['owner']]);sums=np.add.reduceat(exp,data['starts'])
    log_normalizer=maxima+np.log(sums)
    normalized_weight=data['weights'][data['owner']]/data['weights'].sum()
    loss=np.sum(normalized_weight*data['q']*(log_normalizer[data['owner']]-score))+.5*l2*np.dot(w,w)
    residual=normalized_weight*(exp/sums[data['owner']]-data['q'])
    gradient=np.asarray(data['X'].T@residual).ravel()+l2*w
    return float(loss),gradient


def fit(data,l2,check_budget=lambda:None):
    def objective(w):check_budget();return loss_gradient(w,data,l2)
    result=minimize(objective,np.zeros(data['X'].shape[1]),method='L-BFGS-B',jac=True,
        options={'maxiter':500,'ftol':1e-11,'gtol':1e-6,'maxls':40})
    converged=bool(result.success and np.isfinite(result.fun) and np.isfinite(result.x).all())
    return {'weights':result.x,'scale':data['scale'],'converged':converged,
        'info':dict(data['info'],l2=l2,iterations=int(result.nit),objective=float(result.fun),
                    optimizer_message=str(result.message),gradient_inf_norm=float(np.max(abs(result.jac))),
                    initialized_from_zero=True,converged=converged)}


def corrections(model,X,offsets,indices):
    values=np.asarray(X@(model['weights']/model['scale'])).ravel()
    return [values[offsets[i]:offsets[i+1]] for i in indices]


def choices(rows,indices,correction,support,config):
    return [residual_choice([c['count'] for c in rows[i]['candidates']],
             value if support[i] else None,config['lambda'],config['margin'])
            for i,value in zip(indices,correction)]


def outcomes(rows,indices,selected,support):
    out=[]
    for i,pick in zip(indices,selected):
        r=rows[i];base=r['candidates'][0]['smiles'] if r['candidates'] else None
        prediction=r['candidates'][pick]['smiles'] if pick is not None else None
        bc=base==r['target'];correct=prediction==r['target']
        out.append({'source_row_id':r['source_row_id'],'group':r['group'],'fold':r['fold'],
            'baseline_correct':bc,'correct':correct,'changed':prediction!=base,
            'correction':not bc and correct,'new_error':bc and not correct,
            'wrong_to_wrong':not bc and not correct and prediction!=base,
            'choice_index':pick,'prediction':prediction,'feature_supported':bool(support[i]),
            'fallback_unsupported':not support[i],'empty_pool':not bool(r['candidates']),
            'difference':int(correct)-int(bc)})
    return out


def summarize(out):
    n=len(out)
    counts={key:sum(bool(r[key]) for r in out) for key in ('baseline_correct','correct','changed','correction',
        'new_error','wrong_to_wrong','fallback_unsupported','empty_pool')}
    return dict(counts,rows=n,baseline_accuracy=counts['baseline_correct']/n,accuracy=counts['correct']/n,
        paired_change=(counts['correct']-counts['baseline_correct'])/n,intervention_rate=counts['changed']/n)


def grid():
    return [{'l2':10.,'weighting':'uniform','lambda':0.,'margin':0.}]+[
        {'l2':l2,'weighting':weight,'lambda':lam,'margin':margin}
        for l2 in L2_GRID for weight in WEIGHTINGS for lam in LAMBDAS for margin in MARGINS]


def selection_key(result):
    s,c=result['summary'],result['config']
    return (s['correct'],-s['new_error'],-s['changed'],c['l2'],-c['lambda'],c['margin'],c['weighting']=='uniform')


def choose_policy(rows,indices,scores,support):
    reports=[]
    for config in grid():
        key=(config['l2'],config['weighting'])
        if config['lambda'] and key not in scores:continue
        values=scores.get(key,[None]*len(indices))
        picks=choices(rows,indices,values,support,config)
        reports.append({'config':config,'summary':summarize(outcomes(rows,indices,picks,support))})
    return max(reports,key=selection_key),reports


def inner_folds(rows,indices,seed):
    assignments={rows[i]['group']:int(hashlib.sha256((str(seed)+':'+rows[i]['group']).encode()).hexdigest()[:16],16)%4 for i in indices}
    return {i:assignments[rows[i]['group']] for i in indices}


def assert_disjoint(rows,left,right):
    for key in ('group','exact_input','normalized_input'):
        if {rows[i][key] for i in left}&{rows[i][key] for i in right}:raise ValueError('Fold leakage: '+key)


def cross_validated_policy(rows,X,offsets,support,indices,folds,nfold,fit_callback=None):
    """All selection labels are confined to indices; outer labels are never read."""
    scores={(l2,w):[None]*len(indices) for l2 in L2_GRID for w in WEIGHTINGS}
    rejected=set();position={i:j for j,i in enumerate(indices)}
    for fold in range(nfold):
        fitting=[i for i in indices if folds[i]!=fold];held=[i for i in indices if folds[i]==fold]
        if not fitting or not held:raise ValueError('Empty development fold')
        assert_disjoint(rows,fitting,held)
        for weighting in WEIGHTINGS:
            data=fit_inputs(rows,X,offsets,support,fitting,weighting)
            for l2 in L2_GRID:
                fitted=fit_callback(fitting,l2,weighting,fold,data) if fit_callback else fit(data,l2)
                if not fitted['converged']:rejected.add((l2,weighting));continue
                values=corrections(fitted,X,offsets,held)
                for i,value in zip(held,values):scores[(l2,weighting)][position[i]]=value
    for key in rejected:scores.pop(key)
    for values in scores.values():
        if any(v is None for v in values):raise ValueError('Incomplete grouped OOF scores')
    selected,reports=choose_policy(rows,indices,scores,support)
    return selected,reports,sorted(rejected)
