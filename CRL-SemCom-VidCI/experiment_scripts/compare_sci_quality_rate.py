import json
import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm.autonotebook import tqdm

sys.path.append(os.path.abspath(".."))

from src import dataloading, models, summary_utils, utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    utils.seed(seed)


def build_model(args, checkpoint_path, legacy_action_space=False):
    args.legacy_action_space = legacy_action_space
    shutter = models.define_shutter(args.shutter, args, model_dir=args.log_root)
    decoder = models.define_decoder(args.decoder, args)
    model = models.define_model(shutter, decoder, args, get_coded=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def map_actions_to_ratio(actions, legacy_action_space=False):
    actions = actions.float()
    ratio = torch.zeros_like(actions)
    if legacy_action_space:
        ratio = torch.where(actions == -1, 0 * torch.ones_like(ratio), ratio)
    ratio = torch.where(actions == 0, 1 * torch.ones_like(ratio), ratio)
    ratio = torch.where(actions == 1, 2 * torch.ones_like(ratio), ratio)
    ratio = torch.where(actions == 2, 4 * torch.ones_like(ratio), ratio)
    ratio = torch.where(actions == 3, 8 * torch.ones_like(ratio), ratio)
    return ratio


def evaluate_model(model, dataloader, legacy_action_space=False, max_batches=20):
    psnrs = []
    ssims = []
    ratios = []

    with torch.no_grad():
        for batch_idx, (_, model_input, gt, ref_input, _, _) in enumerate(tqdm(dataloader, disable=False)):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            restored, _, actions, _ = model([model_input, ref_input], train=False)
            psnrs.append(float(summary_utils.get_psnr(restored, gt)))
            ssims.append(float(summary_utils.get_ssim(restored, gt)))
            ratio_map = map_actions_to_ratio(actions, legacy_action_space=legacy_action_space)
            ratios.append(float(ratio_map.mean().item()))

    return {
        "num_batches": len(psnrs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "avg_ratio": float(np.mean(ratios)),
    }


def main(args):
    set_seed(args.seed)
    if args.interp == "none":
        args.interp = None

    val_dataloader, _ = dataloading.loadValDataset(args)

    model_a = build_model(args, args.checkpoint_a, legacy_action_space=args.legacy_action_space_a)
    model_b = build_model(args, args.checkpoint_b, legacy_action_space=args.legacy_action_space_b)

    result = {
        "subset_batches": args.max_batches,
        "model_a": {
            "name": args.name_a,
            "checkpoint": args.checkpoint_a,
            "legacy_action_space": args.legacy_action_space_a,
            **evaluate_model(model_a, val_dataloader, legacy_action_space=args.legacy_action_space_a, max_batches=args.max_batches),
        },
        "model_b": {
            "name": args.name_b,
            "checkpoint": args.checkpoint_b,
            "legacy_action_space": args.legacy_action_space_b,
            **evaluate_model(model_b, val_dataloader, legacy_action_space=args.legacy_action_space_b, max_batches=args.max_batches),
        },
    }

    delta_psnr = result["model_b"]["psnr"] - result["model_a"]["psnr"]
    delta_ratio = result["model_b"]["avg_ratio"] - result["model_a"]["avg_ratio"]
    result["delta_b_minus_a"] = {
        "psnr": delta_psnr,
        "avg_ratio": delta_ratio,
    }

    print(json.dumps(result, indent=2))

    if args.output_json != "":
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--dataset_name", type=str, default="nfs_block_rgb_256_8f")
    parser.add_argument("--log_root", type=str, default="../logs")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument("--name_a", type=str, default="model_a")
    parser.add_argument("--name_b", type=str, default="model_b")
    parser.add_argument("--legacy_action_space_a", action="store_true")
    parser.add_argument("--legacy_action_space_b", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("-b", "--block_size", default="8,256,256")
    parser.add_argument("--resume", type=str, default="00-00-00")
    parser.add_argument("--gt", type=int, default=0)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="compare")
    parser.add_argument("--restore_dic_name", type=str, default="MST_fixed")
    parser.add_argument("--reg", type=float, default=100.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--interp", type=str, default="none", choices=["none", "bilinear", "scatter"])
    parser.add_argument("--init", type=str, choices=["even", "ones", "quad"], default="quad")
    parser.add_argument("--loss", type=str, choices=["mpr", "l1", "l2_lpips", "l2"], default="l2")
    parser.add_argument("--decoder", type=str, default="MST")
    parser.add_argument("--shutter", type=str, default="lsvpe")
    parser.add_argument("--sched", type=str, default="reduce")
    args = parser.parse_args()
    args.block_size = [int(item) for item in args.block_size.split(",")]
    main(args)
