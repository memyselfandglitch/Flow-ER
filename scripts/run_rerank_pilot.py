#!/usr/bin/env python3
"""Run a bounded pilot; never append to or resume the paused full benchmark."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from rerank_analysis import analyze_row, canonical


def extract_records(root,pilot,name,selection):
    spec=selection['parts'][name]
    inputs=pilot/'inputs'/f'{name}.txt'
    assert hashlib.sha256(inputs.read_bytes()).hexdigest()==spec['inputs_sha256']
    metadata_path=pilot/f'{name}-rows.json'
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest()==spec['row_manifest_sha256']
    metadata=json.loads(metadata_path.read_text())
    log=(pilot/'logs'/f'inference-{name}.log').read_text(errors='replace')
    skipped=re.findall(r'([\d.]+)% data is unparseable',log)
    assert skipped and all(float(x)==0 for x in skipped),'Cannot prove prediction alignment'
    prediction=pilot/'predictions'/name/'test-64-2940000_97.txt'
    raw=prediction.read_bytes();assert raw.endswith(b'\n')
    lines=raw.decode().splitlines();gt=inputs.read_text().splitlines()
    assert len(lines)==len(gt)==len(metadata)==spec['rows']
    dest=pilot/'records'/f'{name}.jsonl';dest.parent.mkdir(exist_ok=True)
    assert not dest.exists(),'Records already exist; inspect before repeating'
    with dest.open('w') as output:
        for index,(truth,prediction_line,meta) in enumerate(zip(gt,lines,metadata)):
            assert hashlib.sha256((truth+'\n').encode()).hexdigest()==meta['source_line_sha256']
            row=analyze_row(truth,prediction_line,index,10)
            assert row['sample_count']==64
            assert hashlib.sha256(canonical(row['reactant']).encode()).hexdigest()==meta['reactant_key']
            row.update(meta);output.write(json.dumps(row)+'\n')
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=7200)
    args=p.parse_args();root=args.root.resolve();os.chdir(root)
    pilot=root/'results/reranking/pilot-v2';(pilot/'logs').mkdir(exist_ok=True)
    selection=json.loads((pilot/'selection.json').read_text())
    paused=json.loads((root/'.vast/paused-run.json').read_text())
    assert selection['diagnostic_prefix_rows']==paused['completed_rows']
    full_prediction=root/paused['prediction']
    assert hashlib.sha256(full_prediction.read_bytes()).hexdigest()==paused['prediction_sha256']
    assert not (pilot/'pilot-status.json').exists(),'Pilot already started; inspect saved status'
    env=dict(os.environ,PATH='/opt/conda/bin:'+os.environ['PATH'],NUM_WORKERS='8',
             TEST_BATCH_SIZE='2048',MAX_SAMPLE_MATRIX_ELEMENTS='10000000',OMP_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    started=time.time();deadline=started+args.max_seconds
    state={'status':'running','started_at':started,'deadline':deadline,'phase':'setup',
           'full_benchmark_status':'paused','expected_rows':{k:v['rows'] for k,v in selection['parts'].items()}}
    def save():
        temporary=pilot/'pilot-status.json.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(pilot/'pilot-status.json')
    def run(phase,command,extra_env=None):
        state['phase']=phase;state['phase_started_at']=time.time();save()
        remaining=deadline-time.time()
        if remaining<=0: raise TimeoutError('Pilot hard time limit reached')
        with (pilot/'logs'/f'{phase}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=root,env=dict(env,**(extra_env or {})),stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=log,start_new_session=True)
            state['phase_pid']=process.pid;save()
            try: code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('Pilot hard time limit reached')
            if code: raise RuntimeError(f'{phase} exited {code}; inspect its log')
        print('Completed phase:',phase,flush=True)
    save()
    try:
        run('dependencies',[sys.executable,'-m','pip','install','scikit-learn==1.5.2','joblib==1.5.3'])
        xtb=root/'.vast/tools/xtb-dist/bin/xtb'
        assert xtb.is_file(),'Missing verified xTB binary'
        # Fit/tune outcomes are available to development; holdout is not opened
        # until all models and their decision rules have been frozen.
        for name in ['fit','tune']:
            run('inference-'+name,['bash','scripts/test_vast.sh'],
                {'TEST_FILE':str(pilot/'inputs'/f'{name}.txt'),'RESULT_PATH':str(pilot/'predictions'/name)})
            extract_records(root,pilot,name,selection)
        energy_args=['--cache',str(pilot/'energy-cache'),'--xtb',str(xtb),'--solvent','gas','--k','5',
                     '--max-rows','0','--conformers','1','--timeout','20','--max-heavy-atoms','40','--jobs','4']
        run('energy-tune',[sys.executable,'scripts/score_xtb_candidates.py','--records',str(pilot/'records/tune.jsonl'),
                           '--out',str(pilot/'energies-tune.jsonl'),*energy_args])
        run('freeze-model',[sys.executable,'scripts/pilot_reranker.py','fit-tune',
                            '--fit',str(pilot/'records/fit.jsonl'),'--tune',str(pilot/'records/tune.jsonl'),
                            '--tune-energies',str(pilot/'energies-tune.jsonl'),'--k','5',
                            '--holdout-manifest',str(pilot/'holdout-rows.json'),'--out',str(pilot/'model')])
        frozen=json.loads((pilot/'model/frozen.json').read_text());assert frozen['status']=='frozen_before_test_evaluation'
        state['model_frozen_at']=frozen['frozen_at'];save()
        run('inference-holdout',['bash','scripts/test_vast.sh'],
            {'TEST_FILE':str(pilot/'inputs/holdout.txt'),'RESULT_PATH':str(pilot/'predictions/holdout')})
        extract_records(root,pilot,'holdout',selection)
        run('energy-holdout',[sys.executable,'scripts/score_xtb_candidates.py','--records',str(pilot/'records/holdout.jsonl'),
                              '--out',str(pilot/'energies-holdout.jsonl'),*energy_args])
        energy_protocols=[json.loads((pilot/f'energies-{name}.jsonl.manifest.json').read_text())['computation_protocol_sha256']
                          for name in ['tune','holdout']]
        assert len(set(energy_protocols))==1,'Energy protocols differ'
        run('evaluate',[sys.executable,'scripts/pilot_reranker.py','evaluate','--records',str(pilot/'records/holdout.jsonl'),
                        '--model',str(pilot/'model/frozen.joblib'),'--energies',str(pilot/'energies-holdout.jsonl'),
                        '--holdout-manifest',str(pilot/'holdout-rows.json'),'--out',str(pilot/'comparison')])
        assert hashlib.sha256(full_prediction.read_bytes()).hexdigest()==paused['prediction_sha256']
        state.update(status='completed',finished_at=time.time(),full_benchmark_prefix_unchanged=True)
    except Exception as exc:
        state.update(status='failed_or_incomplete',finished_at=time.time(),error=str(exc),error_type=type(exc).__name__)
        print('Pilot stopped:',str(exc),flush=True)
    finally:
        save()
        # Copyable final receipt before stopping GPU billing; retain disk and all
        # artifacts. Full benchmark resumption is a separate gated action.
        files={str(path.relative_to(pilot)):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in pilot.rglob('*') if path.is_file() and path.name!='pilot-artifacts.json'}
        (pilot/'pilot-artifacts.json').write_text(json.dumps(files,indent=2))
        print('Pilot status:',state['status'],'; stopping GPU after backup grace period',flush=True)
        time.sleep(120)
        credentials=json.loads(Path('/root/flower-instance-env.json').read_text())
        instance=credentials.get('CONTAINER_ID') or credentials['VAST_CONTAINERLABEL'].split('.')[-1]
        assert str(instance)=='58056052'
        request=urllib.request.Request(f'https://console.vast.ai/api/v0/instances/{instance}/',
            data=b'{"state":"stopped"}',headers={'Authorization':'Bearer '+credentials['CONTAINER_API_KEY'],
                                                'Content-Type':'application/json'},method='PUT')
        with urllib.request.urlopen(request,timeout=30) as response:
            response.read()


if __name__=='__main__': main()
