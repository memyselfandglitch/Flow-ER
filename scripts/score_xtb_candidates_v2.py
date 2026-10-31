"""Versioned physical adapter. Importing or validating records runs no xTB."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

import score_xtb_candidates as old
import xtb_structure_v2 as structure

VERSION = 'flower_energy_focus_adapter_v2'


def protocol(args):
    return {'version':VERSION, 'method':'GFN2-xTB', 'solvent':args.solvent,
        'reference_state':'bar1M', 'electronic_temperature_K':300.,
        'monatomic':'one single point at origin; no optimizer or duplicate conformers',
        'molecular_conformers':2, 'embedding':'ETKDGv3', 'seed':42,
        'optimization':'normal', 'unpaired_electrons':0, 'threads':1,
        'fragment_wall_seconds':20., 'require_all_conformers':True,
        'max_heavy_atoms':40, 'strict_elements':sorted(old.SUPPORTED_ELEMENTS),
        'structure_policy':structure.VERSION, 'resonance_flags':structure.RESONANCE_FLAGS,
        'resonance_limit':structure.RESONANCE_LIMIT,
        'atom_correspondence':'xTB XYZ atom order; verify original/output ordered elements and full indexed bonds including H',
        'failed_calculation_policy':'never import energies from an originally failed fragment or nonzero/nonconverged calculation',
        'xtb_binary_sha256':args.xtb_sha256, 'rdkit_version':rdBase.rdkitVersion,
        'adapter_sha256':old.file_hash(__file__),
        'structure_checker_sha256':old.file_hash(structure.__file__),
        'aggregation':'sum fragment minima with multiplicities; ALPB already includes solvation'}


def configuration(smiles, args):
    return {'smiles':smiles, 'method':'GFN2-xTB', 'solvent':args.solvent,
            'computation_protocol':protocol(args)}


def command(path, charge, monatomic, args):
    cmd=[args.xtb,str(path),'--gfn','2','--chrg',str(charge),'--uhf','0',
         '--etemp','300.0','--alpb',args.solvent,'bar1M']
    return cmd+(['--sp'] if monatomic else ['--opt','normal'])


def physical_check(item, charge, solvent, monatomic=False):
    """A low energy cannot override an execution, convergence or setting failure."""
    if item.get('returncode') != 0:
        return {'valid':False,'reason':'nonzero_or_missing_returncode'}
    text=item.get('stdout',''); upper=text.upper()
    if ('NORMAL TERMINATION OF XTB' not in upper or
        re.search(r'ABNORMAL TERMINATION|\[ERROR\]|FAILED TO CONVERGE|(?:SCC|SCF)[^\n]*NOT CONVERG',upper)):
        return {'valid':False,'reason':'failed_or_missing_normal_termination'}
    if not re.search(r'CONVERGENCE CRITERIA SATISFIED AFTER \d+ ITERATIONS',upper):
        return {'valid':False,'reason':'missing_scc_convergence'}
    if not monatomic and 'GEOMETRY OPTIMIZATION CONVERGED' not in upper:
        return {'valid':False,'reason':'missing_geometry_convergence'}
    cmd=item.get('command',[])
    def option(flag):
        return cmd[cmd.index(flag)+1] if flag in cmd and cmd.index(flag)+1<len(cmd) else None
    if (option('--gfn')!='2' or option('--chrg')!=str(charge) or option('--uhf')!='0' or
        option('--etemp') not in ('300','300.0') or option('--alpb')!=solvent or
        ('--opt' in cmd if monatomic else option('--opt')!='normal')):
        return {'valid':False,'reason':'command_physical_settings_mismatch'}
    if monatomic and '--sp' not in cmd:
        return {'valid':False,'reason':'monatomic_single_point_required'}
    alpb=cmd.index('--alpb')
    if len(cmd)<=alpb+2 or cmd[alpb+2] not in ('bar1M','bar1mol'):
        return {'valid':False,'reason':'reference_state_argument_mismatch'}
    if (not re.search(r'Solvation model:\s+ALPB',text) or
        not re.search(r'^\s*Solvent\s+'+re.escape(solvent)+r'\s*$',text,re.M) or
        'gsolv [1 bar gas/1 M solution]' not in text):
        return {'valid':False,'reason':'output_solvent_or_reference_state_mismatch'}
    energies=re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)',text)
    if not energies or not math.isfinite(float(energies[-1])):
        return {'valid':False,'reason':'missing_or_nonfinite_total_energy'}
    return {'valid':True,'energy_hartree':float(energies[-1])}


def validate_conformer(smiles, item, solvent, monatomic=False):
    expected=structure.expected_molecule(smiles)
    physical=physical_check(item,Chem.GetFormalCharge(expected),solvent,monatomic)
    if not physical['valid']:
        return {'valid':False,'physical':physical,'reason':physical['reason']}
    result=structure.check_geometry(smiles,item.get('input_xyz',''),item.get('optimized_xyz',''))
    if not result['valid']:
        return {'valid':False,'physical':physical,'structure':result,'reason':result['reason']}
    if not monatomic:
        match=re.search(r'(?:SCF done\s+|energy:\s*)(-?\d+\.\d+)',item.get('optimized_xyz',''))
        if not match or abs(float(match[1])-physical['energy_hartree'])>1e-8:
            return {'valid':False,'reason':'geometry_energy_provenance_mismatch','structure':result}
    return {'valid':True,'energy_hartree':physical['energy_hartree'],'structure':result}


def reuse_decision(record, old_execution):
    # Successful conformers inside a failed fragment are deliberately not salvaged.
    if record.get('status')!='ok':
        return {'reusable':False,'reason':'original_failed_fragment_quarantined'}
    p=record.get('computation_protocol',{})
    required={'protocol':'ETKDGv3_xtb_explicit_solution_complete_conformers_v3',
        'method':'GFN2-xTB','solvent':record['solvent'],'conformers':2,'reference_state':'bar1mol',
        'electronic_temperature_K':300.,'require_all_conformers':True,'strict_elements':True,
        'seed':42,'threads_per_fragment':1,'optimization':'normal','unpaired_electrons':0,
        'timeout_seconds':20.,'max_heavy_atoms':40,'rdkit_version':rdBase.rdkitVersion,
        'xtb_binary_sha256':old_execution['xtb_binary_sha256'],
        'scorer_sha256':old_execution['execution_code_sha256']['scripts/score_xtb_candidates.py'],
        'canonicalization_sha256':old_execution['execution_code_sha256']['scripts/rerank_analysis.py']}
    if any(p.get(k)!=v for k,v in required.items()):
        return {'reusable':False,'reason':'old_protocol_not_compatible'}
    items=record.get('conformer_records',[])
    if len(items)!=2 or {c.get('conformer_id') for c in items}!={0,1} or not record.get('complete_conformers'):
        return {'reusable':False,'reason':'incomplete_original_conformer_set'}
    validations=[validate_conformer(record['smiles'],c,record['solvent']) for c in items]
    if not all(c.get('status')=='ok' and v['valid'] for c,v in zip(items,validations)):
        return {'reusable':False,'reason':'conformer_revalidation_failed','validations':validations}
    es=[v['energy_hartree'] for v in validations]
    if record.get('conformer_energies_hartree')!=es or record.get('energy_hartree')!=min(es) or any(
        c.get('energy_hartree')!=e for c,e in zip(items,es)):
        return {'reusable':False,'reason':'saved_energy_provenance_mismatch'}
    return {'reusable':True,'reason':'original_success_revalidated','validations':validations}


def imported_result(record, decision, args, source_path, source_sha):
    if not decision.get('reusable'):
        raise ValueError('Unvalidated or failed cache record cannot be imported')
    result=copy.deepcopy(record)
    result.update(configuration(record['smiles'],args))
    result['reuse']={'source_path':str(source_path),'source_sha256':source_sha,
                    'source_protocol':record['computation_protocol'],'decision':decision}
    return result


def calculate_fragment(smiles,args):
    unsupported=old.fragment_unsupported(smiles,args)
    if unsupported:
        return {'status':'unsupported','energy_hartree':None,'reason':unsupported}
    mol=structure.expected_molecule(smiles);mono=mol.GetNumAtoms()==1
    if mono:
        conformer=Chem.Conformer(1);conformer.SetAtomPosition(0,(0.,0.,0.));mol.AddConformer(conformer)
        ids=[0]
    else:
        params=AllChem.ETKDGv3();params.randomSeed=42;params.numThreads=1
        ids=list(AllChem.EmbedMultipleConfs(mol,numConfs=2,params=params))
    records=[];energies=[];failures=[]
    for conf in ids:
        item={'conformer_id':int(conf),'status':'failed','energy_hartree':None}
        records.append(item)
        with tempfile.TemporaryDirectory(prefix='flower-xtb-v2-') as folder:
            path=Path(folder)/'input.xyz';Chem.MolToXYZFile(mol,str(path),confId=conf)
            cmd=command(path,Chem.GetFormalCharge(mol),mono,args)
            item.update(command=cmd,input_xyz=path.read_text())
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            try:
                run=subprocess.run(cmd,cwd=folder,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,timeout=20.)
            except subprocess.TimeoutExpired:
                item['reason']='xtb_timeout';failures.append(item['reason']);continue
            item.update(returncode=run.returncode,stdout=run.stdout)
            if (Path(folder)/'NOT_CONVERGED').exists() or (Path(folder)/'.sccnotconverged').exists():
                item['reason']='nonconvergence_marker';failures.append(item['reason']);continue
            if mono:
                item['optimized_xyz']=item['input_xyz']
                item['geometry_role']='unchanged_single_atom_single_point_input; not an optimized geometry'
            elif (Path(folder)/'xtbopt.xyz').exists():
                item['optimized_xyz']=(Path(folder)/'xtbopt.xyz').read_text()
            validation=validate_conformer(smiles,item,args.solvent,mono);item['validation']=validation
            if not validation['valid']:
                item['reason']=validation['reason'];failures.append(item['reason']);continue
            item.update(status='ok',energy_hartree=validation['energy_hartree'],topology_preserved=True)
            energies.append(validation['energy_hartree'])
    needed=1 if mono else 2
    complete=len(energies)==len(ids)==needed
    return {'status':'ok' if complete else 'failed','energy_hartree':min(energies) if complete else None,
        'reason':None if complete else 'incomplete_or_unreliable_conformer_set',
        'calculation_kind':'monatomic_single_point' if mono else 'molecular_optimization',
        'requested_conformers':needed,'attempted_conformers':len(ids),'converged_conformers':len(energies),
        'complete_conformers':complete,'conformer_energies_hartree':energies,
        'conformer_records':records,'failures':failures}


def compute_bounded(smiles,args):
    remaining=min(20.,args.deadline-time.monotonic())
    if remaining<=0:return {'status':'failed','energy_hartree':None,'reason':'total_budget_exhausted'}
    settings={k:getattr(args,k) for k in ('xtb','solvent','max_heavy_atoms','strict_elements')}
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_fragment-worker'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,start_new_session=True)
    try:stdout,stderr=p.communicate(json.dumps({'smiles':smiles,'settings':settings}),timeout=remaining)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.communicate()
        return {'status':'failed','energy_hartree':None,'reason':'fragment_wall_timeout'}
    if p.returncode:
        return {'status':'failed','energy_hartree':None,'reason':'fragment_worker_failed','stderr':stderr}
    return json.loads(stdout)


def fragment_energy(smiles,args):
    config=configuration(smiles,args);path=old.cache_path(config,args)
    # Runtime files are fresh. A matching cached failure is a recorded outcome,
    # never a source of energy and never an automatic retry.
    if path.exists():
        data=json.loads(path.read_text())
        if all(data.get(k)==v for k,v in config.items()) and old.valid_result(data):return data
        raise ValueError('Unexpected or incompatible runtime cache')
    data={**compute_bounded(smiles,args),**config}
    if not old.valid_result(data):data=old.failure(config,'invalid_fragment_result')
    old.atomic_json(path,data);return data


if __name__=='__main__':
    if sys.argv[1:]!=['_fragment-worker']:
        raise SystemExit('Use the manifest-gated run_energy_focus_v2.py after authorization')
    request=json.load(sys.stdin)
    print(json.dumps(calculate_fragment(request['smiles'],SimpleNamespace(**request['settings'])),allow_nan=False))
