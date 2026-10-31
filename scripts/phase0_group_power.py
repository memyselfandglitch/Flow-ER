#!/usr/bin/env python3
"""Paired cluster-bootstrap power simulation using input-only group sizes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from phase0_reranking_audit import save, digest


def bootstrap_ratios(sizes, differences, rng, replicates):
    """Exactly the pairs group bootstrap, compressed by identical (n_g,d_g)."""
    sizes=np.asarray(sizes,dtype=np.int64); differences=np.asarray(differences,dtype=np.int64)
    if len(sizes)!=len(differences) or not len(sizes) or np.any(sizes<=0) or np.any(abs(differences)>sizes):
        raise ValueError('Invalid paired group totals')
    pairs,counts=np.unique(np.column_stack([sizes,differences]),axis=0,return_counts=True)
    draws=rng.multinomial(len(sizes),counts/counts.sum(),size=replicates)
    return (draws@pairs[:,1])/(draws@pairs[:,0])


def sizes_for_n(histogram,n,seed):
    sizes=np.asarray([int(size) for size,count in histogram.items() for _ in range(count)])
    if n==sizes.sum(): return sizes
    rng=np.random.default_rng(seed); result=[]; total=0
    while total<n:
        chunk=rng.choice(sizes,size=max(10,int((n-total)/sizes.mean())+10))
        for size in chunk:
            result.append(int(size));total+=int(size)
            if total>=n: break
    # Keep the final complete group. The actual denominator is recorded.
    return np.asarray(result,dtype=np.int64)


def simulate_differences(sizes,delta,discordance,rho,rng):
    if not 0<=rho<1 or not abs(delta)<=discordance<=1: raise ValueError('Invalid simulation rates')
    p=np.asarray([(discordance+delta)/2,(discordance-delta)/2,1-discordance])
    if rho:
        if np.any(p<=0): raise ValueError('Dirichlet scenario needs positive category probabilities')
        probabilities=rng.dirichlet(p*(1/rho-1),size=len(sizes))
        pr,ph=probabilities[:,0],probabilities[:,1]
    else:
        pr,ph=p[0],p[1]
    rescued=rng.binomial(sizes,pr)
    remaining=sizes-rescued
    conditional=np.divide(ph,1-pr,out=np.zeros_like(np.asarray(ph,dtype=float)),where=np.asarray(1-pr)>0)
    harmed=rng.binomial(remaining,np.clip(conditional,0,1))
    return rescued-harmed, rescued.sum(), harmed.sum()


def wilson(successes,n):
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def scenario(job):
    profile,hist,n,q,rho,delta,trials,boots,seed=job
    binding=json.dumps([profile,n,q,rho,delta,seed])
    rng=np.random.default_rng(int(hashlib.sha256(binding.encode()).hexdigest()[:16],16))
    sizes=sizes_for_n(hist,n,seed+n)
    positive=negative=ruleout=0; observed=[]; widths=[]; rescued=[]; harmed=[]
    for _ in range(trials):
        differences,r,h=simulate_differences(sizes,delta,q,rho,rng)
        estimates=bootstrap_ratios(sizes,differences,rng,boots)
        lo,hi=np.quantile(estimates,[.025,.975])
        positive+=lo>0;negative+=hi<0;ruleout+=hi<.01
        observed.append(differences.sum()/sizes.sum());widths.append(hi-lo)
        rescued.append(r/sizes.sum());harmed.append(h/sizes.sum())
    return {'profile':profile,'nominal_rows':n,'actual_rows':int(sizes.sum()),'groups':len(sizes),
        'delta':delta,'discordance':q,'within_group_rho':rho,'simulations':trials,'bootstrap_replicates':boots,
        'positive_detection_power':positive/trials,'power_mc_95ci':wilson(positive,trials),
        'two_sided_rejection_rate':(positive+negative)/trials,
        'probability_upper_ci_below_1pp':ruleout/trials,
        'mean_effect':float(np.mean(observed)),'mean_rescue_rate':float(np.mean(rescued)),
        'mean_harm_rate':float(np.mean(harmed)),'mean_ci_width_pp':100*float(np.mean(widths)),
        'assumed_design_effect':float(1+rho*(sum(sizes*sizes)/sum(sizes)-1))}


def run(root,workers):
    output=root/'power.json'
    if output.exists(): raise FileExistsError('Power result already exists; do not silently replace it.')
    protocol=json.loads((root/'protocol.json').read_text())['power_design']
    bias=json.loads((root/'bias.json').read_text())
    profiles={'fixed_test_selection':bias['fixed_test_selection']['group_profile']['size_histogram'],
              'validation_sensitivity':bias['validation']['group_profile']['size_histogram']}
    plan={'primary_grid':protocol['n_grid'],'validation_profile_sensitivity_n':[20000],
        'profile_policy':'At 20k use exact frozen-selection group sizes; other N sample whole groups from that profile. Full-validation size profile is a conservative separate sensitivity at 20k.',
        'simulation_count':protocol['simulations'],'bootstrap_replicates':protocol['bootstrap_replicates'],
        'seed':protocol['seed'],'declared_at':time.time(),
        'test_labels_used':False,'script_sha256':digest(Path(__file__))}
    save(root/'power-plan.json',plan)
    jobs=[]
    for name,hist in profiles.items():
        for n in (protocol['n_grid'] if name=='fixed_test_selection' else [20000]):
            for q in protocol['discordance_rates']:
                for rho in protocol['within_group_correlations']:
                    jobs.append((name,hist,n,q,rho,protocol['gain'],protocol['simulations'],protocol['bootstrap_replicates'],protocol['seed']))
        null=protocol['null_check']
        jobs.append((name,hist,null['n'],null['discordance'],null['rho'],0.,protocol['simulations'],protocol['bootstrap_replicates'],protocol['seed']))
    results=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(scenario,job) for job in jobs]
        for future in as_completed(futures):
            value=future.result(); results.append(value)
            save(root/'power-progress.json',{'complete':len(results),'total':len(jobs),'elapsed_seconds':time.time()-start})
            print(json.dumps({k:value[k] for k in ('profile','nominal_rows','discordance','within_group_rho','delta','positive_detection_power')}),flush=True)
    results.sort(key=lambda r:(r['profile'],r['delta'],r['discordance'],r['within_group_rho'],r['nominal_rows']))
    required=[]
    for q in protocol['discordance_rates']:
        for rho in protocol['within_group_correlations']:
            group=[r for r in results if r['profile']=='fixed_test_selection' and r['delta']==.01 and r['discordance']==q and r['within_group_rho']==rho]
            for desired in protocol['target_power']:
                passing=[r for r in group if r['positive_detection_power']>=desired]
                conservative=[r for r in group if r['power_mc_95ci'][0]>=desired]
                required.append({'discordance':q,'rho':rho,'target_power':desired,
                    'first_grid_n_point_estimate':min((r['nominal_rows'] for r in passing),default=None),
                    'first_grid_n_mc_lower_bound':min((r['nominal_rows'] for r in conservative),default=None)})
    save(output,{'scenarios':results,'required_sample_grid':required,'elapsed_seconds':time.time()-start,
        'method':'Paired Dirichlet-multinomial cluster simulation and exact compressed pairs group-percentile bootstrap; all differences use row-weighted totals.',
        'limits':['Sample-size recommendations are grid estimates with Monte Carlo intervals, conditional on scenario assumptions.',
            'The outcome/ICC distribution of a future method and exact test-input array mergers are unknown.',
            'Wrong-to-wrong switches have paired difference zero; intervention rate may exceed discordance.',
            'Simulation bootstrap uses 499 replicates for feasibility; frozen evaluation specifies 9,999.',
            'No validation or reserved test outcomes are used to create simulated paired effects.']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.audit,a.workers)
