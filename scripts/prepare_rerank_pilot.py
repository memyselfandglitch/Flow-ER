#!/usr/bin/env python3
"""Freeze an outcome-blind pilot using official train, validation and test files."""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from multiprocessing import Pool
from pathlib import Path

from rerank_analysis import canonical

GENERIC={'','PM','RS','RC','PC'}
SEED='flower-rerank-pilot-20261028-v1'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def raw_group(source,index,line):
    fields=line.rstrip('\n').split('|')
    sequence=fields[-1] if len(fields)>1 else ''
    return f'{source}:{sequence}' if sequence not in GENERIC else f'{source}:row:{index}'


def row_metadata(item):
    source,index,line=item
    fields=line.rstrip('\n').split('|')
    raw_sequence=fields[-1] if len(fields)>1 else ''
    reactant=fields[0].split('>>',1)[0]
    return {'source_split':source,'source_row_id':index,
            'sequence_id':f'{source}:{raw_sequence}' if raw_sequence not in GENERIC else raw_sequence,
            'raw_sequence_id':raw_sequence,'reactant_key':digest(canonical(reactant)),
            'source_line_sha256':digest(line)}


def read_index(source,path,out,workers,selected_raw_groups=None):
    cache=out/f'{source}-input-index.jsonl'
    if cache.exists():
        return [json.loads(line) for line in cache.open()]
    def inputs():
        with path.open() as f:
            for index,line in enumerate(f):
                if selected_raw_groups is None or raw_group(source,index,line) in selected_raw_groups:
                    yield source,index,line
    rows=[]
    with Pool(workers) as pool,cache.open('w') as output:
        for row in pool.imap(row_metadata,inputs(),chunksize=128):
            rows.append(row);output.write(json.dumps(row)+'\n')
            if len(rows)%25000==0: print(source,'indexed',len(rows),'reactant inputs',flush=True)
    return rows


def joined_groups(rows):
    parent=list(range(len(rows)));seen={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        keys=[('reactant',row['reactant_key'])]
        if row['sequence_id'] not in GENERIC: keys.append(('sequence',row['sequence_id']))
        for key in keys:
            if key in seen: parent[find(i)]=find(seen[key])
            else: seen[key]=i
    minimum={}
    for i,row in enumerate(rows):
        group=find(i)
        minimum[group]=min(minimum.get(group,row['reactant_key']),row['reactant_key'])
    for i,row in enumerate(rows): row['split_group']=digest('component:'+minimum[find(i)])


def select_rows(rows,role,size):
    ordered=sorted(rows,key=lambda row:digest(f'{SEED}:{role}:{row["source_split"]}:{row["source_row_id"]}'))
    if len(ordered)<size: raise ValueError(f'Only {len(ordered)} eligible rows for {role}; need {size}')
    return sorted(ordered[:size],key=lambda row:row['source_row_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--paused-config',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--fit-rows',type=int,default=512);p.add_argument('--tune-rows',type=int,default=256)
    p.add_argument('--holdout-rows',type=int,default=512)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'selection.json').exists(),'Selection is frozen; use existing manifest'
    paused=json.loads(a.paused_config.read_text());prefix=paused['completed_rows']
    sources={part:a.data/(part+'.txt') for part in ['train','val','test']}
    hashes={};counts={};train_groups=set()
    for part,path in sources.items():
        h=hashlib.sha256();n=0
        with path.open('rb') as f:
            for index,raw in enumerate(f):
                h.update(raw);n+=1
                if part=='train': train_groups.add(raw_group(part,index,raw.decode()))
        hashes[part]=h.hexdigest();counts[part]=n
    assert hashes['test']==paused['inputs']['data/flower_new_dataset/test.txt']['sha256']
    selected_raw_groups=set(heapq.nsmallest(1024,train_groups,key=lambda group:digest(SEED+':train-pool:'+group)))
    # Only reactants and source-local sequence IDs are used; target outcomes and
    # saved prediction files are never opened during selection.
    train=read_index('train',sources['train'],a.out,a.workers,selected_raw_groups)
    val=read_index('val',sources['val'],a.out,a.workers)
    test=read_index('test',sources['test'],a.out,a.workers)
    all_rows=train+val+test;joined_groups(all_rows)
    diagnostic_groups={r['split_group'] for r in test if r['source_row_id']<prefix}
    fit=select_rows(train,'fit',a.fit_rows)
    fit_groups={r['split_group'] for r in fit}
    tune=select_rows([r for r in val if r['split_group'] not in fit_groups],'tune',a.tune_rows)
    tune_groups={r['split_group'] for r in tune}
    development_groups={r['split_group'] for r in train+val}
    eligible=[r for r in test if r['source_row_id']>=prefix and r['split_group'] not in diagnostic_groups|development_groups]
    holdout=select_rows(eligible,'holdout',a.holdout_rows)
    parts={'fit':fit,'tune':tune,'holdout':holdout}
    assert not fit_groups & tune_groups
    assert not {r['split_group'] for r in holdout} & (diagnostic_groups|development_groups)
    for name,rows in parts.items():
        by_index={r['source_row_id']:r for r in rows}
        (a.out/'inputs').mkdir(exist_ok=True)
        source=rows[0]['source_split']
        with sources[source].open() as f,(a.out/'inputs'/f'{name}.txt').open('w') as output:
            for index,line in enumerate(f):
                if index in by_index:
                    assert digest(line)==by_index[index]['source_line_sha256']
                    output.write(line)
        for index,row in enumerate(rows): row['row_id']=index
        (a.out/f'{name}-rows.json').write_text(json.dumps(rows,indent=2))
    receipt={'status':'frozen_before_outcome_inspection','seed':SEED,'primary_k':5,'sample_size':64,
             'source_sha256':hashes,'source_rows':counts,'diagnostic_prefix_rows':prefix,
             'diagnostic_prediction_sha256':paused['prediction_sha256'],
             'grouping':'Transitive same canonical reactant OR source-local pathway ID; generic labels ignored',
             'group_universe':{'test':len(test),'val':len(val),'train_candidate_pool':len(train)},
             'train_pool_raw_groups':len(selected_raw_groups),'eligible_test_rows':len(eligible),
             'evaluation_exclusions':'All groups touching existing diagnostic prefix, any validation row, or training candidate pool',
             'selection':'Fixed hash sample of rows within eligible groups; no correctness, rank or energy filtering',
             'parts':{name:{'source_split':rows[0]['source_split'],'rows':len(rows),
                            'groups':len({r['split_group'] for r in rows}),
                            'inputs_sha256':hashlib.sha256((a.out/'inputs'/f'{name}.txt').read_bytes()).hexdigest(),
                            'row_manifest_sha256':hashlib.sha256((a.out/f'{name}-rows.json').read_bytes()).hexdigest()}
                      for name,rows in parts.items()},
             'limitation':'Pilot targets eligible unprocessed test rows; it is not a full benchmark accuracy estimate. Unselected training rows are not reranker fitting data.'}
    (a.out/'selection.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': main()
