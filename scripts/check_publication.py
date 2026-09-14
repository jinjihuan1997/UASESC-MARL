#!/usr/bin/env python3
"""Verify the reviewed publication tree. Reads Git only; never stages or pushes."""
import argparse
import json
from pathlib import PurePosixPath
import subprocess

def git(*args):
    return subprocess.check_output(['git', *args])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='Check index instead of HEAD')
    args = parser.parse_args()
    manifest_path = 'PUBLICATION_MANIFEST.json'
    spec = ':' + manifest_path if args.staged else 'HEAD:' + manifest_path
    manifest = json.loads(git('show', spec))
    expected = {e['path']: e for e in manifest['files']}
    actual = {}
    if args.staged:
        for record in git('ls-files', '--stage', '-z').split(b'\0'):
            if not record:
                continue
            info, path = record.split(b'\t', 1)
            mode, oid, stage = info.decode().split()
            if stage != '0':
                raise SystemExit('Unmerged index entry')
            actual[path.decode()] = (mode, oid)
    else:
        for record in git('ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
            if not record:
                continue
            info, path = record.split(b'\t', 1)
            mode, kind, oid = info.decode().split()
            if kind != 'blob':
                raise SystemExit('Nested Git object is not allowed')
            actual[path.decode()] = (mode, oid)
    failures = []
    if set(actual) != set(expected) | {manifest_path}:
        failures.append({'extra': sorted(set(actual)-set(expected)-{manifest_path}), 'missing': sorted(set(expected)-set(actual))})
    blocked = {'.pt','.pth','.ckpt','.safetensors','.onnx','.engine','.pkl','.pickle','.joblib','.h5','.hdf5','.bin','.npy','.jsonl','.log','.gz','.zip','.mp4','.gif'}
    for path, e in expected.items():
        p = PurePosixPath(path)
        if actual.get(path) != (e['mode'], e['oid']):
            failures.append({'path': path, 'error': 'content or mode differs from reviewed manifest'})
        if p.suffix.lower() in blocked or e['bytes'] > manifest['max_file_bytes']:
            failures.append({'path': path, 'error': 'excluded artifact or oversized file'})
        if p.suffix == '.npz' and e['category'] != 'small_physical_profile':
            failures.append({'path': path, 'error': 'NPZ is not a reviewed physical lookup table'})
        if p.suffix == '.csv' and e['category'] != 'compact_summary_table':
            failures.append({'path': path, 'error': 'CSV is not a reviewed aggregate table'})
        if e['mode'] not in {'100644','100755'} or '..' in p.parts or p.is_absolute():
            failures.append({'path': path, 'error': 'invalid path or file mode'})
    print(json.dumps({'passed': not failures, 'checked_files': len(actual), 'payload_bytes_excluding_manifest': sum(e['bytes'] for e in expected.values()), 'failures': failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
