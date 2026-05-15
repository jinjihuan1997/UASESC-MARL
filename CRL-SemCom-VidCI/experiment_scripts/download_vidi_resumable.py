import argparse
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / 'data' / 'VIDI' / 'metadata' / 'vidi_all.csv'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'data' / 'VIDI' / 'raw'
DEFAULT_LOG_DIR = REPO_ROOT / 'data' / 'VIDI'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Robust resumable downloader for the VIDI dataset.'
    )
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--output_root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--tmp_root', type=Path, default=Path('/tmp/vidi_tmp'))
    parser.add_argument('--log_dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--socket_timeout', type=int, default=30)
    parser.add_argument('--download_timeout', type=int, default=180)
    parser.add_argument('--ffmpeg_timeout', type=int, default=120)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--sleep_seconds', type=int, default=15)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--max_items', type=int, default=0,
                        help='Only process up to this many rows in one pass. 0 means no limit.')
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--cookies', type=Path, default=None)
    parser.add_argument('--cookies_from_browser', type=str, default='',
                        help='Forwarded to yt-dlp --cookies-from-browser, e.g. chrome or firefox.')
    return parser.parse_args()


def canonicalize_columns(df):
    rename_map = {}
    if 'youtube_id' in df.columns:
        rename_map['youtube_id'] = 'video_id'
    if 'time_start' in df.columns:
        rename_map['time_start'] = 'start_time'
    if 'time_end' in df.columns:
        rename_map['time_end'] = 'end_time'
    if 'label' in df.columns:
        rename_map['label'] = 'label_name'
    df = df.rename(columns=rename_map)
    required = {'video_id', 'start_time', 'end_time', 'label_name'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    return df


def clip_basename(row):
    return f"{row.video_id}_{int(row.start_time):06d}_{int(row.end_time):06d}.mp4"


def clip_output_path(output_root, row):
    return output_root / str(row.label_name) / clip_basename(row)


def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_jsonl(path):
    if not path.exists():
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def classify_download_error(log_text):
    text = (log_text or '').lower()
    if 'requested format is not available' in text:
        return 'format_unavailable'
    if 'only images are available for download' in text:
        return 'format_unavailable'
    if 'n challenge solving failed' in text:
        return 'format_unavailable'
    if 'sign in to confirm you’' in text or "sign in to confirm you're not a bot" in text:
        return 'auth_required'
    if 'age-restricted' in text:
        return 'auth_required'
    if 'this video is unavailable' in text:
        return 'unavailable'
    if 'private video' in text:
        return 'unavailable'
    if 'video has been removed' in text:
        return 'unavailable'
    if 'http error 403' in text:
        return 'auth_required'
    if 'timeout' in text:
        return 'temporary'
    return 'temporary'


def build_skip_sets(status_log_path, has_auth):
    permanent_unavailable = set()
    auth_required = set()
    format_unavailable = set()
    for record in load_jsonl(status_log_path):
        if 'video_id' not in record:
            continue
        error_type = record.get('error_type', '')
        if not error_type and record.get('status') in {'download_failed', 'trim_failed'} and 'log' in record:
            error_type = classify_download_error(record.get('log', ''))
        if error_type == 'unavailable':
            permanent_unavailable.add(record['video_id'])
        elif error_type == 'auth_required':
            auth_required.add(record['video_id'])
        elif error_type == 'format_unavailable':
            format_unavailable.add(record['video_id'])
    if has_auth:
        auth_required = set()
    return permanent_unavailable, auth_required, format_unavailable


def run_command(command, timeout):
    try:
        output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        return True, output
    except subprocess.TimeoutExpired as exc:
        return False, f'Timeout after {timeout}s: {exc}'
    except subprocess.CalledProcessError as exc:
        return False, exc.output


def download_source_video(video_id, tmp_root, socket_timeout, retries, download_timeout, cookies=None, cookies_from_browser=''):
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix='vidi_dl_', dir=str(tmp_root)))
    template = tmp_dir / '%(id)s.%(ext)s'
    command = [
        'yt-dlp',
        '--quiet',
        '--no-warnings',
        '--socket-timeout',
        str(socket_timeout),
        '--retries',
        str(retries),
        '--fragment-retries',
        str(retries),
        '-o',
        str(template),
        f'https://www.youtube.com/watch?v={video_id}',
    ]
    if cookies is not None:
        command[1:1] = ['--cookies', str(cookies)]
    elif cookies_from_browser:
        command[1:1] = ['--cookies-from-browser', cookies_from_browser]
    ok, output = run_command(command, timeout=download_timeout)
    if not ok:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, output

    files = sorted(tmp_dir.glob('*'))
    if not files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, 'yt-dlp finished without creating a file'
    return files[0], output


def trim_clip(source_path, dest_path, start_time, end_time, ffmpeg_timeout):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    duration = float(end_time) - float(start_time)
    command = [
        'ffmpeg',
        '-v',
        'error',
        '-y',
        '-i',
        str(source_path),
        '-ss',
        str(float(start_time)),
        '-t',
        str(duration),
        '-c:v',
        'libx264',
        '-c:a',
        'copy',
        str(dest_path),
    ]
    return run_command(command, timeout=ffmpeg_timeout)


def make_clip_record(row, output_path, status, **extra):
    record = {
        'status': status,
        'output_path': str(output_path),
        'video_id': row.video_id,
        'label_name': row.label_name,
        'start_time': int(row.start_time),
        'end_time': int(row.end_time),
    }
    record.update(extra)
    return record


def process_video_group(rows, args, skip_sets):
    rows = list(rows)
    video_id = rows[0].video_id
    permanent_unavailable, auth_required, format_unavailable = skip_sets

    missing_rows = []
    records = []
    for row in rows:
        output_path = clip_output_path(args.output_root, row)
        if output_path.exists():
            records.append(make_clip_record(row, output_path, 'exists'))
        else:
            missing_rows.append((row, output_path))

    if not missing_rows:
        return records

    if video_id in permanent_unavailable:
        for row, output_path in missing_rows:
            records.append(make_clip_record(row, output_path, 'skipped_unavailable', error_type='unavailable'))
        return records

    if video_id in auth_required:
        for row, output_path in missing_rows:
            records.append(make_clip_record(row, output_path, 'skipped_auth_required', error_type='auth_required'))
        return records

    if video_id in format_unavailable:
        for row, output_path in missing_rows:
            records.append(make_clip_record(row, output_path, 'skipped_format_unavailable', error_type='format_unavailable'))
        return records

    source_path = None
    source_parent = None
    try:
        source_path, download_log = download_source_video(
            video_id,
            args.tmp_root,
            args.socket_timeout,
            args.retries,
            args.download_timeout,
            cookies=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
        )
        if source_path is None:
            error_type = classify_download_error(download_log if isinstance(download_log, str) else str(download_log))
            if error_type == 'unavailable':
                permanent_unavailable.add(video_id)
            elif error_type == 'auth_required' and not (args.cookies or args.cookies_from_browser):
                auth_required.add(video_id)
            elif error_type == 'format_unavailable':
                format_unavailable.add(video_id)
            for row, output_path in missing_rows:
                records.append(
                    make_clip_record(
                        row,
                        output_path,
                        'download_failed',
                        error_type=error_type,
                        log=(download_log[-800:] if isinstance(download_log, str) else str(download_log)),
                    )
                )
            return records

        source_parent = source_path.parent
        for row, output_path in missing_rows:
            ok, ffmpeg_log = trim_clip(
                source_path,
                output_path,
                row.start_time,
                row.end_time,
                args.ffmpeg_timeout,
            )
            if not ok:
                records.append(
                    make_clip_record(
                        row,
                        output_path,
                        'trim_failed',
                        error_type='trim_failed',
                        log=(ffmpeg_log[-800:] if isinstance(ffmpeg_log, str) else str(ffmpeg_log)),
                    )
                )
            else:
                records.append(make_clip_record(row, output_path, 'downloaded'))
        return records
    finally:
        if source_parent is not None:
            shutil.rmtree(source_parent, ignore_errors=True)


def count_existing(output_root):
    return len(list(output_root.glob('**/*.mp4')))


def run_pass(df, args, status_log_path):
    processed = 0
    downloaded = 0
    existed = 0
    failed = 0
    skipped = 0

    has_auth = bool(args.cookies or args.cookies_from_browser)
    skip_sets = build_skip_sets(status_log_path, has_auth=has_auth)

    iterable = df.iloc[args.start_index:]
    if args.max_items > 0:
        iterable = iterable.iloc[:args.max_items]

    grouped_rows = []
    groups = defaultdict(list)
    for row in iterable.itertuples(index=False):
        groups[row.video_id].append(row)
    for video_id in groups:
        grouped_rows.append(groups[video_id])

    start_ts = time.time()
    for idx, rows in enumerate(grouped_rows, start=1):
        records = process_video_group(rows, args, skip_sets)
        for record in records:
            append_jsonl(status_log_path, record)
            processed += 1
            if record['status'] == 'downloaded':
                downloaded += 1
            elif record['status'] == 'exists':
                existed += 1
            elif record['status'].startswith('skipped_'):
                skipped += 1
            else:
                failed += 1

        if idx % args.print_every == 0 or any(record['status'] == 'downloaded' for record in records):
            elapsed = time.time() - start_ts
            print(
                f"[{time.strftime('%F %T')}] processed={processed} downloaded={downloaded} "
                f"exists={existed} skipped={skipped} failed={failed} mp4_total={count_existing(args.output_root)} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    return {
        'processed': processed,
        'downloaded': downloaded,
        'exists': existed,
        'skipped': skipped,
        'failed': failed,
        'mp4_total': count_existing(args.output_root),
    }


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    status_log_path = args.log_dir / 'vidi_resumable_status.jsonl'
    pass_log_path = args.log_dir / 'vidi_resumable_passes.jsonl'

    df = canonicalize_columns(pd.read_csv(args.csv))
    print(
        f'Loaded {len(df)} rows from {args.csv}. '
        f'Current mp4 count: {count_existing(args.output_root)}',
        flush=True,
    )

    while True:
        pass_start = {
            'timestamp': time.strftime('%F %T'),
            'event': 'pass_start',
            'current_mp4': count_existing(args.output_root),
        }
        append_jsonl(pass_log_path, pass_start)
        print(f"[{pass_start['timestamp']}] pass_start current_mp4={pass_start['current_mp4']}", flush=True)

        summary = run_pass(df, args, status_log_path)
        pass_end = {
            'timestamp': time.strftime('%F %T'),
            'event': 'pass_end',
            **summary,
        }
        append_jsonl(pass_log_path, pass_end)
        print(
            f"[{pass_end['timestamp']}] pass_end processed={summary['processed']} downloaded={summary['downloaded']} "
            f"exists={summary['exists']} skipped={summary['skipped']} failed={summary['failed']} "
            f"current_mp4={summary['mp4_total']}",
            flush=True,
        )

        if not args.loop:
            break
        time.sleep(args.sleep_seconds)


if __name__ == '__main__':
    main()
