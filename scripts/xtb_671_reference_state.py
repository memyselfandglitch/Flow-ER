#!/usr/bin/env python3
"""Spelling-only bridge from frozen bar1mol convention to xTB 6.7.1 bar1M.

No scientific settings are changed. Bind the wrapper to the inspected binary
so cache provenance cannot silently refer to a different backend.
"""
import hashlib
import os
from pathlib import Path
import sys

BINARY = Path.home()/'.local/share/flower-xtb/bin/xtb'
BINARY_SHA256 = 'd3174941efad80a6862d6d985c70907c23221962067713cb8f8bfbb0f4b471ff'
SOURCE = 'https://github.com/grimme-lab/xtb/blob/v6.7.1/src/prog/main.F90#L1734-L1746'


def translated_arguments(arguments):
    result = list(arguments)
    for i, argument in enumerate(result):
        if argument == '--alpb' and i+2 < len(result) and result[i+2] == 'bar1mol':
            result[i+2] = 'bar1M'
    return result


def main():
    if hashlib.sha256(BINARY.read_bytes()).hexdigest() != BINARY_SHA256:
        raise SystemExit('Inspected xTB binary changed; execution refused')
    if '--version' in sys.argv[1:]:
        print('Frozen bar1mol -> xTB bar1M spelling bridge; backend SHA256 '+BINARY_SHA256, flush=True)
    os.execv(str(BINARY), [str(BINARY)]+translated_arguments(sys.argv[1:]))


if __name__ == '__main__':
    main()
