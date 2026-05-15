import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPTS_DIR = REPO_ROOT / 'experiment_scripts'
DEFAULT_RAW_ROOT = REPO_ROOT / 'data' / 'VIDI' / 'raw'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'data' / 'vidi_block_rgb_256_8f'
FRAMES_PER_CLIP = 32
FRAME_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert downloaded VIDI .mp4 clips into the frame index format used by the SCI/SemCom training code.'
    )
    parser.add_argument('--raw_root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument('--output_root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--frames_per_clip', type=int, default=FRAMES_PER_CLIP)
    parser.add_argument('--frame_size', type=int, default=FRAME_SIZE)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def stable_score(text):
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()
    return int(digest, 16) / float(16 ** len(digest))


def probe_video(video_path):
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=nb_frames,avg_frame_rate,width,height:format=duration',
        '-of',
        'json',
        str(video_path),
    ]
    output = subprocess.check_output(cmd, text=True)
    info = json.loads(output)
    stream = info['streams'][0]
    duration = float(info['format'].get('duration') or 0.0)
    frame_rate = stream.get('avg_frame_rate', '0/1')
    num, den = frame_rate.split('/')
    fps = float(num) / float(den) if float(den) else 0.0
    nb_frames = stream.get('nb_frames')
    if nb_frames and str(nb_frames).isdigit():
        total_frames = int(nb_frames)
    else:
        total_frames = int(round(duration * fps))
    return {
        'width': int(stream.get('width') or 0),
        'height': int(stream.get('height') or 0),
        'duration': duration,
        'fps': fps,
        'total_frames': total_frames,
    }


def choose_frame_indices(total_frames, frames_per_clip):
    if total_frames < frames_per_clip:
        return None
    indices = np.round(np.linspace(0, total_frames - 1, frames_per_clip)).astype(int).tolist()
    if len(set(indices)) != frames_per_clip:
        return None
    return indices


def to_training_path(path):
    rel = path.resolve().relative_to(REPO_ROOT.resolve())
    return str(Path('..') / rel)


def extract_frames(video_path, frame_dir, frames_per_clip, frame_size, force=False):
    expected = [frame_dir / f'{idx:06d}.jpg' for idx in range(frames_per_clip)]
    if not force and all(path.exists() for path in expected):
        return expected

    meta = probe_video(video_path)
    indices = choose_frame_indices(meta['total_frames'], frames_per_clip)
    if indices is None:
        raise RuntimeError(
            f'{video_path} does not have enough distinct frames for {frames_per_clip}-frame extraction '
            f'(total_frames={meta["total_frames"]}).'
        )

    tmp_dir = frame_dir.with_name(f'{frame_dir.name}.tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    expr = '+'.join(f'eq(n\\,{idx})' for idx in indices)
    vf = f'select={expr},scale={frame_size}:{frame_size}'
    cmd = [
        'ffmpeg',
        '-v',
        'error',
        '-y',
        '-i',
        str(video_path),
        '-vf',
        vf,
        '-vsync',
        '0',
        '-q:v',
        '2',
        '-start_number',
        '0',
        str(tmp_dir / '%06d.jpg'),
    ]
    subprocess.run(cmd, check=True)

    frame_paths = sorted(tmp_dir.glob('*.jpg'))
    if len(frame_paths) != frames_per_clip:
        raise RuntimeError(
            f'{video_path} extracted {len(frame_paths)} frames, expected {frames_per_clip}.'
        )

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    tmp_dir.rename(frame_dir)
    return [frame_dir / f'{idx:06d}.jpg' for idx in range(frames_per_clip)]


def collect_videos(raw_root):
    video_paths = sorted(raw_root.glob('**/*.mp4'))
    clips_by_label = {}
    for video_path in video_paths:
        label = video_path.parent.name
        clips_by_label.setdefault(label, []).append(video_path)
    return clips_by_label


def assign_split(video_path, label, test_ratio, seed):
    score = stable_score(f'{seed}:{label}:{video_path.stem}')
    return 'test' if score < test_ratio else 'train'


def enforce_label_split_balance(assignments):
    for label, items in assignments.items():
        if len(items) < 2:
            items[0]['split'] = 'train'
            continue
        splits = {item['split'] for item in items}
        if len(splits) == 1:
            ordered = sorted(items, key=lambda item: stable_score(f'fix:{item["video_path"].stem}'))
            if next(iter(splits)) == 'train':
                ordered[0]['split'] = 'test'
            else:
                ordered[-1]['split'] = 'train'


def build_assignments(raw_root, test_ratio, seed):
    clips_by_label = collect_videos(raw_root)
    assignments = {}
    for label, videos in clips_by_label.items():
        assignments[label] = []
        for video_path in videos:
            assignments[label].append(
                {
                    'label': label,
                    'video_path': video_path,
                    'split': assign_split(video_path, label, test_ratio, seed),
                }
            )
    if assignments:
        enforce_label_split_balance(assignments)
    return assignments


def process_clip(item, output_root, frames_per_clip, frame_size, force=False):
    split = item['split']
    label = item['label']
    video_path = item['video_path']
    frame_dir = output_root / 'frames' / split / label / video_path.stem
    frame_dir.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = extract_frames(
        video_path,
        frame_dir,
        frames_per_clip=frames_per_clip,
        frame_size=frame_size,
        force=force,
    )
    return {
        'split': split,
        'label': label,
        'video_path': str(video_path),
        'frame_paths': [to_training_path(path) for path in frame_paths],
    }


def build_index(processed_items):
    split_records = {'train': {}, 'test': {}}
    split_metadata = {'train': [], 'test': []}
    for split in ['train', 'test']:
        by_label = {}
        for item in processed_items:
            if item['split'] != split:
                continue
            by_label.setdefault(item['label'], []).append(item)
        for vid_num, label in enumerate(sorted(by_label)):
            clips = sorted(by_label[label], key=lambda item: Path(item['video_path']).stem)
            split_records[split][vid_num] = {}
            split_metadata[split].append(
                {
                    'vid_num': vid_num,
                    'label': label,
                    'num_clips': len(clips),
                }
            )
            for clip_num, item in enumerate(clips):
                split_records[split][vid_num][clip_num] = item['frame_paths']
    return split_records, split_metadata


def save_outputs(output_root, split_records, split_metadata, summary):
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'test']:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(split_records[split], split_dir / 'nfs_block_file_locations.pt')
        with open(split_dir / 'vidi_split_metadata.json', 'w') as f:
            json.dump(split_metadata[split], f, indent=2)
    with open(output_root / 'vidi_dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    assignments = build_assignments(args.raw_root, args.test_ratio, args.seed)
    total_candidates = sum(len(items) for items in assignments.values())
    if total_candidates == 0:
        raise SystemExit(f'No .mp4 clips found under {args.raw_root}')

    processed = []
    failures = []
    tasks = [item for items in assignments.values() for item in items]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_to_item = {
            executor.submit(
                process_clip,
                item,
                args.output_root,
                args.frames_per_clip,
                args.frame_size,
                args.force,
            ): item
            for item in tasks
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                processed.append(future.result())
            except Exception as exc:  # pragma: no cover - keeps conversion robust.
                failures.append(
                    {
                        'video_path': str(item['video_path']),
                        'label': item['label'],
                        'split': item['split'],
                        'error': str(exc),
                    }
                )

    split_records, split_metadata = build_index(processed)
    summary = {
        'raw_root': str(args.raw_root),
        'output_root': str(args.output_root),
        'frames_per_clip': args.frames_per_clip,
        'frame_size': args.frame_size,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'total_candidates': total_candidates,
        'processed_clips': len(processed),
        'failed_clips': len(failures),
        'train_videos': len(split_records['train']),
        'test_videos': len(split_records['test']),
        'train_clips': sum(len(v) for v in split_records['train'].values()),
        'test_clips': sum(len(v) for v in split_records['test'].values()),
        'failures': failures,
    }
    save_outputs(args.output_root, split_records, split_metadata, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
