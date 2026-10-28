#!/usr/bin/env bash
# Run inside pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime on the GPU host.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt
python - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

assert torch.cuda.is_available(), 'CUDA is unavailable'
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB')
    assert p.total_memory >= 25 * 2**30, 'FlowER requires at least 25 GiB per GPU'
for filename, expected in json.loads(Path('.vast/manifest.json').read_text()).items():
    p = Path(filename)
    assert p.stat().st_size == expected['bytes'], filename
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['sha256'], filename
print('CUDA and input checksums verified')
PY
