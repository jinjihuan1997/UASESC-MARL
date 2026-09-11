import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm.autonotebook import tqdm


def main(args):
    sys.path.append("/home/king/Downloads/Projects/TON/CRL-SemCom-VidCI")

    from experiment_scripts.comm.metrics import CommunicationMetricAccumulator
    from src import dataloading, models, summary_utils, utils

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    utils.seed(args.seed)

    args.use_semantic_comm = True
    if args.interp == "none":
        args.interp = None

    shutter = models.define_shutter(args.shutter, args, model_dir=args.log_root)
    decoder = models.define_decoder(args.decoder, args)
    model = models.define_model(shutter, decoder, args, get_coded=False).cuda()
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if key in current_state and current_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)

    val_data = dataloading.loadValDataset(args)
    if val_data is None:
        raise RuntimeError("Validation dataset is unavailable")
    val_dataloader, _ = val_data

    loss_fn = utils.define_loss(args)
    val_losses = []
    val_psnrs = []
    val_ssims = []
    comm_metrics = CommunicationMetricAccumulator()

    with torch.no_grad():
        for batch_idx, (_, model_input, gt, ref_input, _, _) in enumerate(tqdm(val_dataloader)):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            model_input = model_input.cuda()
            gt = gt.cuda()
            ref_input = ref_input.cuda()
            restored, _, _, extra = model(
                [model_input, ref_input],
                train=False,
                forced_rate_level=args.eval_forced_rate_level,
            )

            val_losses.append(float(loss_fn(restored, gt).detach().cpu().item()))
            val_psnrs.append(float(summary_utils.get_psnr(restored, gt)))
            val_ssims.append(float(summary_utils.get_ssim(restored, gt)))
            comm_metrics.update(extra["comm"])

    comm = comm_metrics.summary()
    summary = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "mu_comm": args.mu_comm,
        "comm_snr_db": args.comm_snr_db,
        "eval_forced_rate_level": args.eval_forced_rate_level,
        "num_batches": len(val_psnrs),
        "loss": float(np.mean(val_losses)),
        "psnr": float(np.mean(val_psnrs)),
        "ssim": float(np.mean(val_ssims)),
        "avg_rate_level": float(comm["avg_rate_level"]),
        "avg_kept_real_symbols": float(comm["avg_kept_real_symbols"]),
        "bar_ls_main": float(comm["bar_ls_main"]),
        "avg_ls_side": float(comm["avg_ls_side"]),
        "bar_ls_total": float(comm["bar_ls_total"]),
        "bar_ls_paper": float(comm["bar_ls_paper"]),
    }

    print(json.dumps(summary, indent=2))

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        exists = os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="")
    parser.add_argument("--data_root", default="/home/king/Downloads/Projects/TON/CRL-SemCom-VidCI/data")
    parser.add_argument("--dataset_name", default="nfs_block_rgb_256_8f")
    parser.add_argument("--log_root", default="/home/king/Downloads/Projects/TON/CRL-SemCom-VidCI/logs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("-b", "--block_size", default="8,256,256")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", default="00-00-00")
    parser.add_argument("--exp_name", default="quick_eval")
    parser.add_argument("--restore_dic_name", default="MST_fixed")
    parser.add_argument("--gt", type=int, default=0)
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--reg", type=float, default=100.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--interp", default="none", choices=["none", "bilinear", "scatter"])
    parser.add_argument("--init", choices=["even", "ones", "quad"], default="quad")
    parser.add_argument("--loss", choices=["mpr", "l1", "l2_lpips", "l2"], default="l2")
    parser.add_argument("--decoder", default="MST")
    parser.add_argument("--shutter", default="lsvpe")
    parser.add_argument("--sched", default="reduce")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mu_comm", type=float, default=1e-3)
    parser.add_argument("--comm_sensor_channels", type=int, default=24)
    parser.add_argument("--comm_latent_channels", type=int, default=48)
    parser.add_argument("--comm_hidden_channels", type=int, default=64)
    parser.add_argument("--comm_rate_levels", type=int, default=4)
    parser.add_argument("--comm_channel_coding_rate", type=float, default=0.5)
    parser.add_argument("--comm_modulation_order", type=int, default=4)
    parser.add_argument("--comm_snr_db", type=float, default=10.0)
    parser.add_argument("--comm_forced_rate_level", type=int, default=None)
    parser.add_argument("--eval_forced_rate_level", type=int, default=None)
    args = parser.parse_args()
    args.block_size = [int(item) for item in args.block_size.split(",")]
    main(args)
