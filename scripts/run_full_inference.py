#!/usr/bin/env python3
"""Run the existing resumable evaluator with an explicitly bound GNU MKL runtime."""
import os
from pathlib import Path

os.environ['MKL_THREADING_LAYER'] = 'GNU'
import run_targeted_inference as base

original_configuration = base.configuration
original_identity = base.collect_identity


def configuration(args):
    value = original_configuration(args)
    value['environment']['MKL_THREADING_LAYER'] = 'GNU'
    return value


def identity(*args):
    value = original_identity(*args)
    value['code_sha256']['scripts/run_full_inference.py'] = base.sha256(Path(__file__))
    return value


if __name__ == '__main__':
    base.configuration = configuration
    base.collect_identity = identity
    base.main()
