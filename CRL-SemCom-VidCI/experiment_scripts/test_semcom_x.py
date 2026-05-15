import csv
import json
import os
import random
import sys

sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm

from experiment_scripts.comm.metrics import CommunicationMetricAccumulator
from src import dataloading, models, summary_utils, utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    utils.seed(seed)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint['model_state_dict'].items()
        if key in current_state and current_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)


def maybe_write_json(path, payload):
    if path == '':
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=4)


def maybe_append_csv(path, row):
    if path == '':
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    set_seed(args.seed)
    args.use_semantic_comm = True
    if args.interp == 'none':
        args.interp = None

    shutter = models.define_shutter(args.shutter, args, model_dir=args.log_root)
    decoder = models.define_decoder(args.decoder, args)
    model = models.define_model(shutter, decoder, args, get_coded=False)
    model.cuda()
    model.eval()

    if args.checkpoint == '':
        raise ValueError('--checkpoint is required')
    load_checkpoint(model, args.checkpoint)

    val_data = dataloading.loadValDataset(args)
    if val_data is None:
        raise ValueError('Validation dataset is unavailable when --test is set')
    val_dataloader, _ = val_data

    loss_fn = utils.define_loss(args)
    val_losses = []
    val_psnrs = []
    val_ssims = []
    comm_metric_accumulator = CommunicationMetricAccumulator()

    sample_rows = []

    with torch.no_grad():
        for batch_idx, (_, model_input, gt, ref_input, vid_num, clip_num) in enumerate(
            tqdm(val_dataloader, disable=False)
        ):
            model_input = model_input.cuda()
            gt = gt.cuda()
            ref_input = ref_input.cuda()

            restored, _, _, extra = model(
                [model_input, ref_input],
                train=False,
                forced_rate_level=args.eval_forced_rate_level,
            )

            val_loss = loss_fn(restored, gt)
            psnr = summary_utils.get_psnr(restored, gt)
            ssim = summary_utils.get_ssim(restored, gt)
            comm_info = extra['comm']
            comm_metric_accumulator.update(comm_info)

            val_losses.append(float(val_loss.detach().cpu().item()))
            val_psnrs.append(float(psnr))
            val_ssims.append(float(ssim))

            if batch_idx < args.max_logged_batches:
                sample_rows.append(
                    {
                        'batch_idx': batch_idx,
                        'vid_num': int(vid_num[0].detach().cpu().item()),
                        'clip_num': int(clip_num[0].detach().cpu().item()),
                        'psnr': float(psnr),
                        'ssim': float(ssim),
                        'avg_rate_level': float(comm_info['avg_rate_level']),
                        'avg_kept_real_symbols': float(comm_info['avg_kept_real_symbols']),
                        'avg_ls_main': float(comm_info['avg_ls_main']),
                        'avg_ls_side': float(comm_info['avg_ls_side']),
                        'avg_ls_total': float(comm_info['avg_ls_total']),
                    }
                )

    comm_summary = comm_metric_accumulator.summary()
    summary = {
        'checkpoint': args.checkpoint,
        'mu_comm': args.mu_comm,
        'comm_snr_db': args.comm_snr_db,
        'eval_forced_rate_level': args.eval_forced_rate_level,
        'num_val_batches': len(val_psnrs),
        'loss': float(np.mean(val_losses)),
        'psnr': float(np.mean(val_psnrs)),
        'ssim': float(np.mean(val_ssims)),
        'avg_rate_level': float(comm_summary['avg_rate_level']),
        'avg_kept_real_symbols': float(comm_summary['avg_kept_real_symbols']),
        'bar_ls_main': float(comm_summary['bar_ls_main']),
        'avg_ls_side': float(comm_summary['avg_ls_side']),
        'bar_ls_total': float(comm_summary['bar_ls_total']),
        'bar_ls_paper': float(comm_summary['bar_ls_paper']),
        'sample_batches': sample_rows,
    }

    print(json.dumps(summary, indent=2))

    csv_row = {
        'checkpoint': args.checkpoint,
        'mu_comm': args.mu_comm,
        'comm_snr_db': args.comm_snr_db,
        'eval_forced_rate_level': args.eval_forced_rate_level,
        'loss': summary['loss'],
        'psnr': summary['psnr'],
        'ssim': summary['ssim'],
        'avg_rate_level': summary['avg_rate_level'],
        'avg_kept_real_symbols': summary['avg_kept_real_symbols'],
        'bar_ls_main': summary['bar_ls_main'],
        'avg_ls_side': summary['avg_ls_side'],
        'bar_ls_total': summary['bar_ls_total'],
        'bar_ls_paper': summary['bar_ls_paper'],
    }
    maybe_write_json(args.output_json, summary)
    maybe_append_csv(args.output_csv, csv_row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str, default='nfs_block_rgb_256_8f')
    parser.add_argument('--log_root', type=str, default='../logs')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--output_csv', type=str, default='')
    parser.add_argument('--max_logged_batches', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-b', '--block_size', default='8,256,256')
    parser.add_argument('--resume', type=str, default='00-00-00')
    parser.add_argument('--gt', type=int, default=0)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--scale', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='SemCom')
    parser.add_argument('--restore_dic_name', type=str, default='MST_fixed')
    parser.add_argument('--reg', type=float, default=100.0)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--interp', type=str, default='none', choices=['none', 'bilinear', 'scatter'])
    parser.add_argument('--init', type=str, choices=['even', 'ones', 'quad'], default='quad')
    parser.add_argument('--loss', type=str, choices=['mpr', 'l1', 'l2_lpips', 'l2'], default='l2')
    parser.add_argument('--decoder', type=str, default='MST')
    parser.add_argument('--shutter', type=str, default='lsvpe')
    parser.add_argument('--sched', type=str, default='reduce')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mu_comm', type=float, default=1e-3)
    parser.add_argument('--comm_sensor_channels', type=int, default=24)
    parser.add_argument('--comm_latent_channels', type=int, default=48)
    parser.add_argument('--comm_hidden_channels', type=int, default=64)
    parser.add_argument('--comm_rate_levels', type=int, default=4)
    parser.add_argument('--comm_channel_coding_rate', type=float, default=0.5)
    parser.add_argument('--comm_modulation_order', type=int, default=4)
    parser.add_argument('--comm_snr_db', type=float, default=10.0)
    parser.add_argument('--comm_forced_rate_level', type=int, default=None)
    parser.add_argument('--eval_forced_rate_level', type=int, default=None)
    args = parser.parse_args()
    args.block_size = [int(item) for item in args.block_size.split(',')]
    main(args)
