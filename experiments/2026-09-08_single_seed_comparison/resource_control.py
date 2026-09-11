"""Reversible Linux process pause; preserves in-memory training state."""
import argparse
import os
from pathlib import Path
import signal
import time
from queue_supervisor import process_info, identity_matches, live, read, write, stamp


def descendants(parents):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item = process_info(int(path.name))
            if item and item['uid'] == os.getuid() and item['ppid'] in parents:
                result.append(item)
    return result


def pause(pids, journal):
    journal = Path(journal)
    if journal.exists():
        raise FileExistsError('Preserving existing pause journal')
    record = dict(state='pausing', created_utc=stamp(), roots=pids, processes=[])
    seen = set()
    pending = [process_info(pid) for pid in pids]
    if any(p is None or p['uid'] != os.getuid() or p['pid'] == os.getpid() for p in pending):
        raise RuntimeError('Pause roots must be live owned processes other than this command')
    write(journal, record)
    try:
        while pending:
            level = []
            for item in pending:
                if item['pid'] in seen or not live(item):
                    continue
                seen.add(item['pid'])
                item['was_stopped'] = item['state'] in ('T', 't')
                record['processes'].append(item)
                write(journal, record)  # Identity recorded before each signal.
                if not item['was_stopped'] and identity_matches(item):
                    os.kill(item['pid'], signal.SIGSTOP)
                level.append(item['pid'])
            # Stopped parents cannot create further children once acknowledged.
            deadline = time.monotonic() + 5
            for item in record['processes']:
                while live(item) and process_info(item['pid'])['state'] not in ('T', 't'):
                    if time.monotonic() > deadline:
                        raise RuntimeError('Could not stop a process promptly')
                    time.sleep(.01)
            pending = descendants(set(level))
        record.update(state='paused', paused_utc=stamp())
        write(journal, record)
        return record
    except BaseException:
        resume(journal)
        raise


def resume(journal, defer_pids=()):
    record = read(journal)
    if record['state'] == 'resumed':
        return record
    missing = []
    for item in reversed(record['processes']):
        if item['pid'] in defer_pids or item['was_stopped']:
            continue
        if live(item):
            os.kill(item['pid'], signal.SIGCONT)
        else:
            missing.append(item['pid'])
    record.update(state='partially_resumed' if defer_pids else 'resumed',
                  resumed_utc=stamp(), deferred_pids=list(defer_pids), missing_pids=missing)
    write(journal, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pause', 'resume'])
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--pids', type=int, nargs='+')
    args = parser.parse_args()
    if args.action == 'pause':
        if not args.pids:
            parser.error('--pids is required for pause')
        result = pause(args.pids, args.journal)
    else:
        result = resume(args.journal)
    print(dict(state=result['state'], process_count=len(result['processes']), journal=str(args.journal.resolve())))


if __name__ == '__main__':
    main()
