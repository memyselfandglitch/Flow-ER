#!/usr/bin/env python3
"""Self-stop this one Vast instance at a fixed deadline, independent of SSH."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def stop_own_instance(credentials_path):
    credentials = json.loads(Path(credentials_path).read_text())
    identity = credentials.get('CONTAINER_ID') or credentials['VAST_CONTAINERLABEL'].split('.')[-1]
    request = urllib.request.Request(
        f'https://console.vast.ai/api/v0/instances/{int(identity)}/',
        data=b'{"state":"stopped"}', method='PUT',
        headers={'Authorization': 'Bearer ' + credentials['CONTAINER_API_KEY'],
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--credentials', default='/root/flower-instance-env.json')
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({'status': 'armed', 'deadline': args.deadline,
                                      'pid': os.getpid(), 'armed_at': time.time()}))
    while time.time() < args.deadline:
        time.sleep(min(30, max(0, args.deadline-time.time())))
    for _ in range(20):
        try:
            stop_own_instance(args.credentials)
            return
        except Exception as exc:
            print(type(exc).__name__, flush=True)
            time.sleep(5)
    raise RuntimeError('Instance deadline stop failed; local guard remains responsible')


if __name__ == '__main__':
    main()
