import json
import os
import random
import sys
import time

sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm

from experiment_scripts.comm.metrics import CommunicationMetricAccumulator
from src import dataloading, models, summary_utils, utils


DEFAULT_SEED = 2023
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)
torch.cuda.manual_seed(DEFAULT_SEED)
torch.cuda.manual_seed_all(DEFAULT_SEED)

utils.seed(123)
torch.set_num_threads(2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    utils.seed(seed)


def freeze_module(module):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def build_semcom_reward(restored, gt, comm_cost_map, mu, pool_size, reward_eps):
    """Build the optimization-time communication reward.

    `comm_cost_map` is intentionally the internal optimization proxy, not the
    paper's l_s metric. The paper-aligned communication metrics are computed
    separately from `rate_mask` and logged under explicit names.
    """
    pixel_error = (restored - gt).pow(2).mean(dim=1, keepdim=True)
    pooled_error = F.avg_pool2d(pixel_error.detach(), kernel_size=pool_size, stride=pool_size)
    return torch.log(1.0 / (pooled_error + reward_eps)) - mu * comm_cost_map.detach()


def load_base_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint['model_state_dict'].items()
        if key in current_state and current_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)


def main(args):
    set_seed(args.seed)
    args.use_semantic_comm = True
    args.exp_name = args.exp_name or 'SemCom'
    if args.test:
        args.steps_til_summary = min(args.steps_til_summary, 10)
        dir_name = f'{args.log_root}/test'
        os.makedirs(dir_name, exist_ok=True)
    else:
        args, dir_name, _ = utils.modify_args(args)
    model_dir, _ = utils.make_model_dir(dir_name, args.test, args.exp_name)

    shutter = models.define_shutter(args.shutter, args, model_dir=model_dir)
    decoder = models.define_decoder(args.decoder, args)
    model = models.define_model(shutter, decoder, args, get_coded=False)
    model.cuda()

    if args.base_checkpoint != '':
        load_base_checkpoint(model, args.base_checkpoint)

    if args.freeze_shutter:
        freeze_module(model.shutter)
    if args.freeze_decoder:
        freeze_module(model.decoder)

    summaries_dir, checkpoints_dir = utils.make_subdirs(model_dir)
    _ = summaries_dir
    with open(f'{dir_name}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    train_dataloader = dataloading.loadTrainingDataset(args)
    val_data = dataloading.loadValDataset(args)
    if val_data is None:
        val_dataloader = None
    else:
        val_dataloader, _ = val_data

    base_params = []
    ran_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'transmission.ran' in name:
            ran_params.append(param)
        else:
            base_params.append(param)

    optim = torch.optim.Adam(
        base_params,
        lr=float(args.mlr),
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    optim_ran = torch.optim.SGD(
        ran_params,
        lr=float(args.rlr),
        momentum=args.ran_momentum,
        weight_decay=args.ran_weight_decay,
    )
    scheduler = utils.define_schedule(optim)
    loss_fn = utils.define_loss(args)

    total_time_start = time.time()
    total_steps = 0

    with tqdm(total=len(train_dataloader) * args.max_epochs) as pbar:
        for epoch in range(args.max_epochs):
            model.train()
            for _, model_input, gt, ref_input, _, _ in train_dataloader:
                model_input = model_input.cuda()
                gt = gt.cuda()
                ref_input = ref_input.cuda()

                optim.zero_grad(set_to_none=True)
                optim_ran.zero_grad(set_to_none=True)

                forced_rate_level = args.comm_forced_rate_level
                if epoch < args.warmup_epochs:
                    forced_rate_level = args.warmup_rate_level

                restored, _, _, extra = model(
                    [model_input, ref_input],
                    train=True,
                    steps=total_steps,
                    forced_rate_level=forced_rate_level,
                )
                recon_loss = loss_fn(restored, gt)

                comm_info = extra['comm']
                reward = build_semcom_reward(
                    restored,
                    gt,
                    comm_info['reward_rate_map'],
                    args.mu_comm,
                    args.reward_pool_size,
                    args.reward_eps,
                )
                if comm_info['log_prob'] is None:
                    rl_loss = recon_loss.new_zeros(())
                else:
                    rl_loss = -(reward * comm_info['log_prob']).mean()
                loss = recon_loss + args.ran_loss_weight * rl_loss

                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
                optim_ran.step()

                total_steps += 1
                pbar.update(1)

                if total_steps % args.steps_til_summary == 0:
                    tqdm.write(
                        "Epoch %d, Loss %0.6f, Recon %0.6f, RL %0.6f, "
                        "AvgRateLevel %0.4f, AvgKeptRealSymbols %0.1f, AvgLsPaper %0.1f, total time %0.2f min"
                        % (
                            epoch,
                            float(loss.detach()),
                            float(recon_loss.detach()),
                            float(rl_loss.detach()),
                            float(comm_info['avg_rate_level']),
                            float(comm_info['avg_kept_real_symbols']),
                            float(comm_info['avg_ls_total']),
                            (time.time() - total_time_start) / 60.0,
                        )
                    )

                    if val_dataloader is not None and total_steps % args.val_interval == 0:
                        with torch.no_grad():
                            model.eval()
                            val_psnrs = []
                            val_ssims = []
                            val_comm_metrics = CommunicationMetricAccumulator()
                            for _, model_input, gt, ref_input, _, _ in tqdm(val_dataloader, disable=True):
                                model_input = model_input.cuda()
                                gt = gt.cuda()
                                ref_input = ref_input.cuda()
                                restored, _, _, extra = model(
                                    [model_input, ref_input],
                                    train=False,
                                    forced_rate_level=args.eval_forced_rate_level,
                                )
                                val_psnrs.append(summary_utils.get_psnr(restored, gt))
                                val_ssims.append(summary_utils.get_ssim(restored, gt))
                                val_comm_metrics.update(extra['comm'])

                            if total_steps % args.save_interval == 0:
                                utils.save_chkpt(model, optim, optim_ran, checkpoints_dir, epoch=epoch, final=False)
                            val_comm_summary = val_comm_metrics.summary()
                            print(
                                "PSNR: %s, SSIM: %s, AvgRateLevel: %s, AvgKeptRealSymbols: %s, bar_l_s: %s"
                                % (
                                    np.mean(val_psnrs),
                                    np.mean(val_ssims),
                                    val_comm_summary['avg_rate_level'],
                                    val_comm_summary['avg_kept_real_symbols'],
                                    val_comm_summary['bar_ls_paper'],
                                )
                            )
                            scheduler.step(torch.tensor(-np.mean(val_psnrs)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str, default='nfs_block_rgb_256_8f')
    parser.add_argument('--log_root', type=str, default='../logs')
    parser.add_argument('--log_group_dir', type=str, default='')
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
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--mlr', type=str, default='5e-5')
    parser.add_argument('--rlr', type=str, default='5e-3')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--ran_weight_decay', type=float, default=0.0)
    parser.add_argument('--ran_momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--steps_til_summary', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--interp', type=str, default='none', choices=['none', 'bilinear', 'scatter'])
    parser.add_argument('--init', type=str, choices=['even', 'ones', 'quad'], default='quad')
    parser.add_argument('--loss', type=str, choices=['mpr', 'l1', 'l2_lpips', 'l2'], default='l2')
    parser.add_argument('--decoder', type=str, default='MST')
    parser.add_argument('--shutter', type=str, default='lsvpe')
    parser.add_argument('--sched', type=str, default='reduce')
    parser.add_argument('--base_checkpoint', type=str, default='')
    parser.add_argument('--freeze_shutter', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    parser.add_argument('--mu_comm', type=float, default=1e-3)
    parser.add_argument('--ran_loss_weight', type=float, default=1.0)
    parser.add_argument('--reward_pool_size', type=int, default=8)
    parser.add_argument('--reward_eps', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_rate_level', type=int, default=4)
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
