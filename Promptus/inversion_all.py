import cv2 as cv
import os
import torch
import numpy as np
import argparse
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from lossbuilder import LossBuilder
from quantization import QParam, FakeQuantize
from diffusers import AutoencoderTiny

# ... (保持原有的 VERSION2SPECS, SubstepSampler, seeded_randn, SeededNoise 类不变) ...
VERSION2SPECS = {
    "SDXL-Turbo-fp16": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
    },
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def inversion(
        model,
        sampler,
        decoder,
        rank,
        interval,
        frame_path,
        max_id,
        H=512,
        W=512,
        seed=0,
        filter=None
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        uc = None
        rand_noise = seeded_randn(shape, seed)
        sigma = torch.Tensor([0.05]).float().cuda()

        # Set the loss functions.
        mse_loss = torch.nn.MSELoss()
        builder = LossBuilder('cuda')
        content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                          ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                          ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                          ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                          ('conv_16', 1)]
        vgg_model, lpips_nodes = builder.get_style_and_content_loss(dict(content_layers))

        # ==============================================================================
        # [修改点 1] 生成包含最后一帧的训练列表 (Adaptive Keyframe)
        # ==============================================================================
        # 生成标准间隔列表
        indices = list(range(0, max_id, interval))

        # 检查是否包含最后一帧 (max_id - 1)。如果不包含，强制加入。
        # 注意：max_id 是总帧数 (例如 60)，有效索引是 0 到 59
        last_valid_idx = max_id - 1
        if indices[-1] != last_valid_idx:
            print(f"[Adaptive Keyframe] Standard interval {interval} ends at {indices[-1]}. "
                  f"Forcing last frame {last_valid_idx} into schedule.")
            indices.append(last_valid_idx)

        print(f"Inversion Schedule (Frame IDs): {indices}")

        # 使用 indices 列表进行循环
        for i, f_id in enumerate(indices):

            prompt_path = os.path.join(frame_path, 'results/rank{}_interval{}/'.format(rank, interval))
            if not os.path.exists(prompt_path):
                os.makedirs(prompt_path)

            # ==========================================================================
            # [修改点 2] 检查是否存在，如果存在则跳过训练 (Skip Existing)
            # ==========================================================================
            target_prompt_file = os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(f_id))

            if os.path.exists(target_prompt_file):
                print(f"[SKIP] Frame {f_id} already processed. Found: {target_prompt_file}")

                # 特殊处理：如果是首帧且被跳过，需要确保 init.pth 存在，否则后续训练会崩
                if f_id == 0:
                    init_pth = os.path.join(prompt_path, 'init.pth')
                    if not os.path.exists(init_pth):
                        print(f"[WARNING] Frame 0 skipped but init.pth not found! This might crash later frames.")
                continue  # 跳过本次循环，不执行后续训练代码

            # 如果没跳过，开始训练
            print(f"[TRAIN] Processing frame {f_id}...")

            # 初始化变量
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            Quant_Param_U = QParam(num_bits=8)
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            Quant_Param_V = QParam(num_bits=8)

            lr = 0.1
            optimizer = torch.optim.Adam([U, V], lr=lr)

            min_loss = 1e9
            latest_min_loss = 1e9

            log_path = os.path.join(prompt_path, '{:05d}'.format(f_id))
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_output = open(os.path.join(log_path, 'log.txt'), 'a')

            # ==========================================================================
            # [修改点 3] 动态处理上一帧逻辑 (处理可变间隔)
            # ==========================================================================
            if i > 0:
                prev_f_id = indices[i - 1]
                # 计算当前段的实际间隔 (可能是 interval=10，也可能是结尾的 interval=4)
                current_interval = f_id - prev_f_id

                # 加载上一帧的 checkpoint
                prev_ckpt_path = os.path.join(prompt_path, '{:05d}/ckpt.pth'.format(prev_f_id))
                if not os.path.exists(prev_ckpt_path):
                    # 如果上一帧被跳过了，checkpoint 应该也是存在的。
                    # 如果报错，说明文件丢失或者逻辑断裂
                    raise RuntimeError(f"Previous checkpoint not found at {prev_ckpt_path}. Cannot resume chain.")

                ckpt_prev = torch.load(prev_ckpt_path)
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
                prev_frame = ckpt_prev["z"]

                total_iterations = 1500
                lr_schedule_cnt = 20
                # 根据“实际间隔”生成步数列表
                step_list_base = [_ for _ in range(1, current_interval + 1)]
            else:
                # 首帧逻辑
                current_interval = 0
                prev_frame = model.encode_first_stage(load_img(os.path.join(frame_path, '00000.png')))
                torch.save(prev_frame, os.path.join(prompt_path, 'init.pth'))
                total_iterations = 10000
                lr_schedule_cnt = 300
                step_list_base = [0]

            # 加噪
            randn = (prev_frame * sigma + rand_noise * (1 - sigma)).detach()
            step_list = step_list_base

            for iter in range(total_iterations):
                loss_list = {}
                for step in step_list:
                    Quant_Param_U.update(U)
                    Q_U = FakeQuantize.apply(U, Quant_Param_U)
                    Quant_Param_V.update(V)
                    Q_V = FakeQuantize.apply(V, Quant_Param_V)

                    if i > 0:
                        # 使用实际间隔 current_interval 进行插值计算
                        factor = 1 / current_interval
                        u = (1 - step * factor) * U_prev + (step * factor) * Q_U
                        v = (1 - step * factor) * V_prev + (step * factor) * Q_V
                        c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)

                        # 计算当前正在优化的中间帧绝对 ID
                        cur_id = prev_f_id + step
                    else:
                        c = (Q_U @ Q_V / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = 0

                    c = {'crossattn': c}

                    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                    samples_x = decoder(samples_z)

                    gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(cur_id)))
                    gt.requires_grad = True

                    vgg_model(torch.cat([gt, samples_x], dim=0))
                    lpips_loss = 0
                    for node in lpips_nodes:
                        lpips_loss += node.loss
                    lpips_loss = lpips_loss / (len(lpips_nodes) + 1e-9)

                    loss = 0.2 * lpips_loss + 0.8 * mse_loss(samples_x, gt)
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu

                    # 减少打印频率以保持控制台整洁
                    if iter % 100 == 0:
                        print('iter: {}, cur_id: {}, loss: {:.4f}'.format(iter, cur_id, loss.item()))

                    if iter % 500 == 0:
                        log_output.write('iter: {}, cur_id: {}, loss: {}\n'.format(iter, cur_id, loss.item()))

                    if iter % 500 == 0:
                        img = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                        img = (255 * img).to(dtype=torch.uint8).permute(0, 2, 3, 1).detach().cpu().numpy()
                        img = img[0][:, :, ::-1]
                        cv.imwrite(os.path.join(log_path, 'iter_{:05d}_id_{:05d}.png'.format(iter, cur_id)), img)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.model.zero_grad()

                    if cur_id not in loss_list.keys():
                        loss_list[cur_id] = loss.item()

                mean_loss = np.mean(list(loss_list.values()))

                if mean_loss < min_loss:
                    min_loss = mean_loss
                    ckpt = {
                        'U': Q_U, 'U_scale': Quant_Param_U.scale, 'U_zero_point': Quant_Param_U.zero_point,
                        'U_bits': Quant_Param_U.num_bits,
                        'V': Q_V, 'V_scale': Quant_Param_V.scale, 'V_zero_point': Quant_Param_V.zero_point,
                        'V_bits': Quant_Param_V.num_bits,
                        'z': samples_z, 'randn': randn, 'iter': iter, 'loss': mean_loss,
                    }
                    torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))

                    U_Byte = Quant_Param_U.quantize_tensor(Q_U).byte()
                    V_Byte = Quant_Param_V.quantize_tensor(Q_V).byte()
                    prompt = {
                        'U': U_Byte, 'V': V_Byte,
                        'U_scale': Quant_Param_U.scale, 'U_zero_point': Quant_Param_U.zero_point,
                        'V_scale': Quant_Param_V.scale, 'V_zero_point': Quant_Param_V.zero_point,
                    }
                    torch.save(prompt, target_prompt_file)

                if i > 0:
                    # 动态训练：分配更多步数给效果最差的帧
                    # 注意：worst_step_offset 是基于 current_interval 的偏移量
                    worst_step_offset = max(loss_list, key=loss_list.get) - prev_f_id
                    step_list = step_list_base + [worst_step_offset] * 2
                    step_list = sorted(step_list)

                lr_schedule_cnt -= 1
                if lr_schedule_cnt == 0:
                    if min_loss == latest_min_loss:
                        lr = max(lr * 0.5, 0.001)
                        optimizer = torch.optim.Adam([U, V], lr=lr)
                        print('reduce lr to: {}'.format(optimizer.param_groups[0]['lr']))
                    latest_min_loss = min_loss
                    lr_schedule_cnt = 20 if i > 0 else 300

            log_output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky")
    parser.add_argument('-max_id', type=int, default=60)  # 建议改为视频总帧数，例如 60
    parser.add_argument('-rank', type=int, default=8)
    parser.add_argument('-interval', type=int, default=10)
    args = parser.parse_args()

    # Set up and load the models.
    version_dict = VERSION2SPECS['SD-Turbo']
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    load_model(model)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).cuda()
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    seed_ = 88
    sampler.noise_sampler = SeededNoise(seed=seed_)

    # Inversion: from video to prompts
    inversion(
        model, sampler, decoder=taesd.decoder, rank=args.rank, interval=args.interval, frame_path=args.frame_path,
        max_id=args.max_id, H=512, W=512, seed=seed_, filter=state.get("filter")
    )