"""Inference-only, map-free features for the fresh validation residual model."""
from functools import lru_cache
import math

import numpy as np
from scipy import sparse
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

import targeted_reranker as chemistry

VERSION='map_free_residual_features_v1'
FP_SIZE=512
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=FP_SIZE)
ELEMENTS=(6,7,8,15,16)
PROPS=('abs_charge','charged_atoms','fragments','radicals','rings','aromatic_atoms')
EXTRA_NAMES=[f'H_attached_to_Z{z}_charge_{q}' for z in ELEMENTS for q in ('negative','zero','positive')]+[
    'heavy_single_bonds','heavy_double_bonds','heavy_triple_bonds','heavy_aromatic_bonds','other_heavy_bonds']
SOLVENTS={'water':'O','methanol':'CO','ethanol':'CCO','THF':'C1CCOC1','DMF':'CN(C)C=O',
    'DMSO':'CS(C)=O','acetonitrile':'CC#N','dichloromethane':'ClCCl','toluene':'Cc1ccccc1',
    'benzene':'c1ccccc1','diethyl_ether':'CCOCC','ethyl_acetate':'CCOC(C)=O','acetone':'CC(C)=O'}
SOLVENT_KEYS={k:Chem.MolToSmiles(Chem.MolFromSmiles(v)) for k,v in SOLVENTS.items()}
NUMERIC_NAMES=(['log_sample_fraction','sample_fraction','retained_pool_fraction','gap_from_frequency_winner',
    'gap_to_next_frequency','input_similarity','frequency_winner_similarity','mean_pool_similarity',
    'no_change','log_fraction_x_pool_entropy','no_change_x_pool_diversity','no_change_x_frequency_gap']+
    [f'{kind}_{name}' for kind in ('candidate','delta','absolute_delta') for name in PROPS]+
    [f'{kind}_{name}' for kind in ('candidate','delta','absolute_delta') for name in chemistry.GROUPS]+
    [f'{kind}_warning_{name}' for kind in ('candidate','delta') for name in chemistry.FLAGS]+
    [f'{kind}_{name}' for kind in ('delta','absolute_delta') for name in EXTRA_NAMES]+
    [f'{kind}_x_input_{name}' for kind in ('changed','no_change') for name in chemistry.CONTEXT_NAMES]+
    [f'input_{name}_present_x_{term}' for name in SOLVENTS for term in ('no_change','delta_abs_charge','delta_attached_H')])
FEATURE_NAMES=NUMERIC_NAMES+[
    f'{kind}_Morgan_r2_bucket_{i}' for kind in ('signed_change','absolute_change','input_x_no_change') for i in range(FP_SIZE)]


@lru_cache(maxsize=60000)
def describe(smiles):
    d=chemistry.describe(smiles)
    if not d['supported']:return d
    with rdBase.BlockLogs():
        mol=Chem.MolFromSmiles(smiles,chemistry.PARSER)
        Chem.SanitizeMol(mol)
        for a in mol.GetAtoms():a.SetAtomMapNum(0)
        extra=np.zeros(len(EXTRA_NAMES))
        for a in mol.GetAtoms():
            if a.GetAtomicNum() in ELEMENTS:
                q=0 if a.GetFormalCharge()<0 else 2 if a.GetFormalCharge()>0 else 1
                extra[3*ELEMENTS.index(a.GetAtomicNum())+q]+=a.GetTotalNumHs(includeNeighbors=True)
        for b in mol.GetBonds():
            if b.GetBeginAtom().GetAtomicNum()==1 or b.GetEndAtom().GetAtomicNum()==1:continue
            v=b.GetBondTypeAsDouble();i={1.:0,2.:1,3.:2,1.5:3}.get(v,4);extra[15+i]+=1
        fp=FP.GetCountFingerprint(mol).GetNonzeroElements()
        parts={Chem.MolToSmiles(Chem.RemoveHs(part),isomericSmiles=False) for part in Chem.GetMolFrags(mol,asMols=True)}
        solvents=np.asarray([float(s in parts) for s in SOLVENT_KEYS.values()])
        return dict(d,extra=extra,count_fp=fp,solvents=solvents)


def inference_view(row):
    # Explicit whitelist: reference labels, mappings, IDs and old derived features
    # cannot enter the chemistry/scoring layer.
    return {'reactant':row['reactant'],'sample_count':row['sample_count'],
        'candidates':[{'smiles':c['smiles'],'count':c['count']} for c in row['candidates']]}


def row_features(view):
    counts=np.asarray([c['count'] for c in view['candidates']],dtype=float)
    n=len(counts)
    if not n:return sparse.csr_matrix((0,len(FEATURE_NAMES))),{'supported':True,'reason':'empty_pool'}
    if view['sample_count']!=64 or np.any(counts<=0) or np.any(counts[:-1]<counts[1:]) or counts.sum()>64:
        raise ValueError('Original 64-sample candidate order/counts required')
    try:
        inp=describe(view['reactant']);ds=[describe(c['smiles']) for c in view['candidates']]
        if not inp['supported'] or not all(d['supported'] for d in ds):
            return sparse.csr_matrix((n,len(FEATURE_NAMES))),{'supported':False,'reason':'parse_or_sanitize_or_descriptor_failure'}
        p=counts/counts.sum();entropy=float(-sum(p*np.log(p)))
        diversity=1-float(np.mean([DataStructs.TanimotoSimilarity(a['fp'],b['fp']) for a in ds for b in ds]))
        numeric=[];ri=[];ci=[];values=[]
        for i,(d,count) in enumerate(zip(ds,counts)):
            no=float(d['canonical']==inp['canonical']);logp=math.log(count/64.)
            v=[logp,count/64.,count/counts.sum(),(count-counts[0])/64.,
               (count-(counts[i+1] if i+1<n else 0))/64.,
               DataStructs.TanimotoSimilarity(d['fp'],inp['fp']),DataStructs.TanimotoSimilarity(d['fp'],ds[0]['fp']),
               np.mean([DataStructs.TanimotoSimilarity(d['fp'],x['fp']) for x in ds]),no,
               logp*entropy,no*diversity,no*(counts[0]-(counts[1] if n>1 else 0))/64.]
            for field,names,kinds in [('properties',PROPS,3),('counts',chemistry.GROUPS,3),('flags',chemistry.FLAGS,2)]:
                a=np.asarray([d[field][k] for k in names]);b=np.asarray([inp[field][k] for k in names])
                v.extend(a);v.extend(a-b)
                if kinds==3:v.extend(abs(a-b))
            delta=d['extra']-inp['extra'];v.extend(delta);v.extend(abs(delta))
            v.extend((1-no)*inp['context']);v.extend(no*inp['context'])
            for present in inp['solvents']:
                v.extend([present*no,present*(d['properties']['abs_charge']-inp['properties']['abs_charge']),present*float(abs(delta[:15]).sum())])
            numeric.append(v)
            diffs={k:d['count_fp'].get(k,0)-inp['count_fp'].get(k,0) for k in d['count_fp'].keys()|inp['count_fp'].keys()}
            for k,value in diffs.items():
                if value:
                    ri.extend([i,i]);ci.extend([len(NUMERIC_NAMES)+k,len(NUMERIC_NAMES)+FP_SIZE+k]);values.extend([value,abs(value)])
            if no:
                for k,value in inp['count_fp'].items():
                    ri.append(i);ci.append(len(NUMERIC_NAMES)+2*FP_SIZE+k);values.append(value)
        dense=np.asarray(numeric,dtype=float)
        if dense.shape!=(n,len(NUMERIC_NAMES)) or not np.isfinite(dense).all():raise ValueError('Feature schema mismatch')
        matrix=sparse.csr_matrix((values,(ri,ci)),shape=(n,len(FEATURE_NAMES)))
        matrix+=sparse.hstack([sparse.csr_matrix(dense),sparse.csr_matrix((n,3*FP_SIZE))],format='csr')
        return matrix,{'supported':True,'reason':'supported_map_free','mapped_edits_used':False}
    except Exception as exc:
        return sparse.csr_matrix((n,len(FEATURE_NAMES))),{'supported':False,'reason':'descriptor_exception:'+type(exc).__name__}


def build(rows,progress=lambda x:None):
    matrices=[];support=[];offsets=[0]
    for i,row in enumerate(rows):
        matrix,info=row_features(inference_view(row));matrices.append(matrix);support.append(info)
        offsets.append(offsets[-1]+matrix.shape[0])
        if (i+1)%1000==0:progress(f'Features: {i+1}/{len(rows)} validation rows')
    return sparse.vstack(matrices,format='csr'),np.asarray(offsets,dtype=np.int64),support
