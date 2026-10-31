#!/usr/bin/env python3
"""Inspect execution holds. This preparation bundle does not launch later phases."""
import argparse
import json
from pathlib import Path


def readiness(root,phase):
    protocol=json.loads((root/'protocol.json').read_text())
    required={1:['reranker_training'],2:['physical_calculations'],
              3:['new_inference','reserved_test_outcomes']}[phase]
    return {'phase':phase,'status':'held','execution_supported':False,
        'holds':{k:protocol['execution_holds'][k] for k in required},
        'reason':'User explicitly holds execution. Later phases require explicit authorization and their reviewed implementation.',
        'plan':str(root/'IMPLEMENTATION_PLAN.md')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--phase',type=int,choices=(1,2,3),required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    a=p.parse_args();result=readiness(a.experiment,a.phase)
    if a.execute: raise SystemExit(result['reason'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
