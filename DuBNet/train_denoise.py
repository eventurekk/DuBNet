import os
import random
import cv2
import torch
import torch.nn as nn
import yaml
import losses
from dataset.data import get_training_data, get_validation_data
from models.DuBNet import Illum_YCRCB_Denoise_IN
from utils import network_parameters
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import utils
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import lpips
from losses import SSIM_Loss
import argparse

# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    action="store_true",
    help="从 model_latest.pth 继续训练（断点续训）"
)
parser.add_argument(
    "--resume_path",
    type=str,
    default="",
    help="可选：指定 checkpoint 路径，不填则默认用 model_dir/model_latest.pth"
)
args = parser.parse_args()


# =========================
# Set Seeds
# =========================
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


# =========================
# GPU
# =========================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# Model
# =========================
print("==> Build the model")
model_restored = Illum_YCRCB_Denoise_IN()
p_number = network_parameters(model_restored)


model_restored.cuda()


# =========================
# Load yaml configuration file
# =========================
with open("training.yaml", "r") as config:
    opt = yaml.safe_load(config)
Train = opt["TRAINING"]
OPT = opt["OPTIM"]


# =========================
# Paths
# =========================
mode = opt["MODEL"]["MODE"]
model_dir = os.path.join(Train["SAVE_DIR"], mode, "models")
utils.mkdir(model_dir)
result_dir = os.path.join(Train["SAVE_DIR"], mode, "results")
utils.mkdir(result_dir)
train_dir = Train["TRAIN_DIR"]
val_dir = Train["VAL_DIR"]

# # =========================
# # FLOPs (student only, exclude KD adapters)
# # =========================
# class _FlopsWrap(nn.Module):
#     """Wrap your net to return a single Tensor for FLOPs tools."""
#     def __init__(self, net: nn.Module):
#         super().__init__()
#         self.net = net
#
#     def forward(self, x):
#         # END版 forward: restored_y, restored_uv, restored
#         _, _, out = self.net(x)
#         return out
#
#
# flops_str = "N/A"
# flops_h = int(Train["TRAIN_PS"])   # 你也可以固定写 256
# flops_w = int(Train["TRAIN_PS"])
#
# try:
#     # Option 1: fvcore (preferred in papers)
#     from fvcore.nn import FlopCountAnalysis
#     _wrap = _FlopsWrap(model_restored).cuda().eval()
#     _dummy = torch.randn(1, 3, flops_h, flops_w).cuda()
#     with torch.no_grad():
#         _flops = FlopCountAnalysis(_wrap, _dummy).total()
#     flops_str = f"{_flops/1e9:.2f} G"
# except Exception as e_fv:
#     try:
#         # Option 2: thop (returns MACs; convert to FLOPs by 2*MACs)
#         from thop import profile
#         _wrap = _FlopsWrap(model_restored).cuda().eval()
#         _dummy = torch.randn(1, 3, flops_h, flops_w).cuda()
#         with torch.no_grad():
#             macs, _ = profile(_wrap, inputs=(_dummy,), verbose=False)
#         flops_str = f"{(2.0*macs)/1e9:.2f} G"
#     except Exception as e_th:
#         print(f"[Warn] FLOPs calc failed. fvcore: {e_fv} | thop: {e_th}")



# =========================
# Logs (每次启动新建一套日志：resume 也一样)
# =========================
log_dir = os.path.join(Train["SAVE_DIR"], mode, "log")
utils.mkdir(log_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

train_log = os.path.join(log_dir, f"train_{timestamp}.csv")
df_train = pd.DataFrame(columns=["epoch", "train Loss", "val PSNR", "val SSIM", "val LPIPS"])
df_train.to_csv(train_log, mode="w", index=False)

loss_log = os.path.join(log_dir, f"loss_{timestamp}.csv")
df_loss = pd.DataFrame(columns=[
    "epoch",
    "train Loss",
    "loss_y",
    "loss_uv",
    "loss_yuv",
    "loss_percept",
    "loss_freq",
    "loss_color"
])
df_loss.to_csv(loss_log, mode="w", index=False)


# =========================
# Optimizer
# =========================
new_lr = float(OPT["LR_INITIAL"])
optimizer = optim.Adam(
    model_restored.parameters(),
    lr=new_lr,
    betas=(0.9, 0.999),
    eps=1e-8,
)


# =========================
# Scheduler
# =========================
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    OPT["EPOCHS"] - warmup_epochs,
    eta_min=float(OPT["LR_MIN"])
)
scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=1,
    total_epoch=warmup_epochs,
    after_scheduler=scheduler_cosine
)

# =========================
# Training state vars (关键：先定义，resume 时覆盖；fresh 时保持默认)
# =========================
start_epoch = 1
best_psnr = 0.0
best_ssim = 0.0
best_lpips = 1e9
best_epoch_psnr = 0
best_epoch_ssim = 0
best_epoch_lpips = 0
last_ssim_val = None
last_lpips_val = None


# =========================
# Resume training (只有 --resume 才触发)
# =========================
if args.resume:
    resume_path = args.resume_path.strip()
    if resume_path == "":
        resume_path = os.path.join(model_dir, "model_latest.pth")

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"[Resume] 找不到 checkpoint：{resume_path}")

    print(f"==> [Resume] Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cuda")

    # 1) 恢复模型与优化器
    model_restored.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # 有些情况下 optimizer state 在 CPU，保险起见搬到 GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(non_blocking=True)

    # 2) 恢复 epoch
    ckpt_epoch = int(ckpt.get("epoch", 0))
    start_epoch = ckpt_epoch + 1
    print(f"==> [Resume] Checkpoint epoch: {ckpt_epoch}, start from epoch: {start_epoch}")

    # 3) 恢复 scheduler（优先用state_dict；旧ckpt无scheduler则快进对齐）
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"==> [Resume] Scheduler loaded. LR={scheduler.get_last_lr()[0]:.6e}")
    else:
        # 兼容旧checkpoint：没有scheduler信息
        # 你原始逻辑：启动前先 scheduler.step() 一次；每个epoch结束再 scheduler.step()
        # 因此 epoch=ckpt_epoch 的checkpoint保存时，scheduler已 step 过 (1 + ckpt_epoch) 次
        scheduler.step()  # 初始那一次
        for _ in range(ckpt_epoch):
            scheduler.step()
        print(f"==> [Resume] Scheduler fast-forward done. LR={scheduler.get_last_lr()[0]:.6e}")

    # 4) 恢复 best / cache（非常重要，否则续训后 best 会从头计、SSIM/LPIPS复用会失效）
    best_psnr = float(ckpt.get("best_psnr", 0.0))
    best_ssim = float(ckpt.get("best_ssim", 0.0))
    best_lpips = float(ckpt.get("best_lpips", 1e9))
    best_epoch_psnr = int(ckpt.get("best_epoch_psnr", 0))
    best_epoch_ssim = int(ckpt.get("best_epoch_ssim", 0))
    best_epoch_lpips = int(ckpt.get("best_epoch_lpips", 0))
    last_ssim_val = ckpt.get("last_ssim_val", None)
    last_lpips_val = ckpt.get("last_lpips_val", None)

else:
    print("==> [Fresh] Start training from scratch (ignore existing checkpoints).")
    # fresh 训练保持原来的行为：训练开始前先 step 一次
    scheduler.step()


# =========================
# Loss
# =========================
char_criterion = losses.CharbonnierLoss()
ssim_criterion = losses.SSIM_Loss()

percept_criterion = losses.PerceptualLoss().cuda()
freq_criterion = losses.FocalFrequencyLoss().cuda()
color_criterion = losses.ColorLoss().cuda()


# =========================
# DataLoaders（方案A：验证支持 batch>1）
# =========================
print("==> Loading datasets")

train_dataset = get_training_data(train_dir, {"patch_size": Train["TRAIN_PS"]})
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=OPT["BATCH"],
    shuffle=True,
    num_workers=64,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False
)

val_dataset = get_validation_data(val_dir, {"patch_size": Train["VAL_PS"]})
VAL_BS = 8  # 按显存调整：8/4/2 都可以
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=VAL_BS,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False
)


# =========================
# Show the training configuration
# =========================
print(f"""==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'] + 1)}
    Batch sizes:        {OPT['BATCH']}
    Val batch size:     {VAL_BS}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                GPU0
------------------------------------------------------------------""")
# Model FLOPs:        {flops_str} @ {flops_h}x{flops_w}

# =========================
# Start training
# =========================
print("==> Training start: ")
total_start_time = time.time()

val_ssim_criterion = SSIM_Loss().cuda()
val_lpips_criterion = lpips.LPIPS(net="alex").cuda()

for epoch in range(start_epoch, OPT["EPOCHS"] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    model_restored.train()

    # 记录各分量 loss 的累积，后面求 epoch 平均
    loss_y_sum = 0.0
    loss_uv_sum = 0.0
    loss_yuv_sum = 0.0
    loss_percept_sum = 0.0
    loss_freq_sum = 0.0
    loss_color_sum = 0.0
    batch_count = 0

    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        target = data[0].cuda(non_blocking=True)
        input_ = data[1].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        target_y, target_uv = target[:, 0, :, :].unsqueeze(1), target[:, 1:3, :, :]
        restored_y, restored_uv, restored = model_restored(input_)

        # Compute loss
        loss_y = char_criterion(restored_y, target_y) + ssim_criterion(restored_y, target_y)
        loss_uv = char_criterion(restored_uv, target_uv) + ssim_criterion(restored_uv, target_uv)
        loss_yuv = char_criterion(restored, target) + ssim_criterion(restored, target)

        loss_yuv_1 = percept_criterion(restored, target)
        loss_yuv_2 = freq_criterion(restored, target)
        loss_yuv_3 = color_criterion(restored, target)

        loss = (
            0.01 * loss_y
            + 0.1 * loss_uv
            + loss_yuv
            + 0.08 * loss_yuv_1
            + 0.04 * loss_yuv_2
            + 0.03 * loss_yuv_3
        )

        # Back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_restored.parameters(), max_norm=3.0)
        optimizer.step()

        # 累积
        epoch_loss += loss.item()
        loss_y_sum += loss_y.item()
        loss_uv_sum += loss_uv.item()
        loss_yuv_sum += loss_yuv.item()
        loss_percept_sum += loss_yuv_1.item()
        loss_freq_sum += loss_yuv_2.item()
        loss_color_sum += loss_yuv_3.item()
        batch_count += 1

    # epoch 平均
    avg_loss = epoch_loss / max(batch_count, 1)
    avg_loss_y = loss_y_sum / max(batch_count, 1)
    avg_loss_uv = loss_uv_sum / max(batch_count, 1)
    avg_loss_yuv = loss_yuv_sum / max(batch_count, 1)
    avg_loss_percept = loss_percept_sum / max(batch_count, 1)
    avg_loss_freq = loss_freq_sum / max(batch_count, 1)
    avg_loss_color = loss_color_sum / max(batch_count, 1)

    # =========================
    # Validation
    # =========================
    if epoch % Train["VAL_AFTER_EVERY"] == 0:
        model_restored.eval()
        TARGET_SIZE = (256, 256)

        psnr_val_rgb_list = []
        ssim_val_rgb_list = []
        lpips_val_rgb_list = []

        for _, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda(non_blocking=True)
            input_ = data_val[1].cuda(non_blocking=True)

            with torch.no_grad():
                _, _, restored = model_restored(input_)

            restored_bhwc = restored.permute(0, 2, 3, 1).detach().cpu().numpy()
            target_bhwc = target.permute(0, 2, 3, 1).detach().cpu().numpy()

            B = restored_bhwc.shape[0]
            for b in range(B):
                restored_np = cv2.cvtColor(restored_bhwc[b].astype(np.float32), cv2.COLOR_YCrCb2BGR)
                target_np = cv2.cvtColor(target_bhwc[b].astype(np.float32), cv2.COLOR_YCrCb2BGR)

                restored_np = np.clip(restored_np, 0.0, 255.0)
                target_np = np.clip(target_np, 0.0, 255.0)

                psnr_val_rgb_list.append(utils.PSNR(target_np, restored_np))

                restored_rgb = cv2.cvtColor(restored_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                target_rgb = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                restored_rgb = np.clip(restored_rgb, 0.0, 1.0)
                target_rgb = np.clip(target_rgb, 0.0, 1.0)

                restored_tensor = torch.from_numpy(restored_rgb).permute(2, 0, 1).unsqueeze(0).cuda().float()
                target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0).cuda().float()

                with torch.no_grad():
                    ssim_loss = val_ssim_criterion(restored_tensor, target_tensor)
                ssim_val_rgb_list.append(1.0 - float(ssim_loss.item()))

                restored_rgb_resized = cv2.resize(restored_rgb, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                target_rgb_resized = cv2.resize(target_rgb, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

                restored_lpips = torch.from_numpy(restored_rgb_resized).permute(2, 0, 1).unsqueeze(0).cuda().float() * 2.0 - 1.0
                target_lpips = torch.from_numpy(target_rgb_resized).permute(2, 0, 1).unsqueeze(0).cuda().float() * 2.0 - 1.0

                with torch.no_grad():
                    lpips_score = float(val_lpips_criterion(restored_lpips, target_lpips).item())
                lpips_score = float(np.clip(lpips_score, 0.0, 1.0))
                lpips_val_rgb_list.append(lpips_score)

        psnr_val_rgb = float(np.mean(psnr_val_rgb_list)) if len(psnr_val_rgb_list) > 0 else 0.0
        ssim_val_rgb = float(np.mean(ssim_val_rgb_list)) if len(ssim_val_rgb_list) > 0 else 0.0
        lpips_val_rgb = float(np.mean(lpips_val_rgb_list)) if len(lpips_val_rgb_list) > 0 else 1.0
        current_lr = scheduler.get_last_lr()[0]

        last_ssim_val = ssim_val_rgb
        last_lpips_val = lpips_val_rgb

        state = {
            "epoch": epoch,
            "state_dict": model_restored.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_lpips": best_lpips,
            "best_epoch_psnr": best_epoch_psnr,
            "best_epoch_ssim": best_epoch_ssim,
            "best_epoch_lpips": best_epoch_lpips,
            "last_ssim_val": last_ssim_val,
            "last_lpips_val": last_lpips_val,
        }

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            state["best_psnr"] = best_psnr
            state["best_epoch_psnr"] = best_epoch_psnr
            torch.save(state, os.path.join(model_dir, "model_bestPSNR.pth"))

        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            state["best_ssim"] = best_ssim
            state["best_epoch_ssim"] = best_epoch_ssim
            torch.save(state, os.path.join(model_dir, "model_bestSSIM.pth"))

        if lpips_val_rgb < best_lpips:
            best_lpips = lpips_val_rgb
            best_epoch_lpips = epoch
            state["best_lpips"] = best_lpips
            state["best_epoch_lpips"] = best_epoch_lpips
            torch.save(state, os.path.join(model_dir, "model_bestLPIPS.pth"))

        print("[epoch %d PSNR: %.4f SSIM: %.4f LPIPS: %.4f --- best_epoch_psnr %d Best_PSNR %.4f | best_epoch_ssim %d Best_SSIM %.4f | best_epoch_lpips %d Best_LPIPS %.4f | LR: %.6f]" %
              (epoch, psnr_val_rgb, ssim_val_rgb, lpips_val_rgb,
               best_epoch_psnr, best_psnr,
               best_epoch_ssim, best_ssim,
               best_epoch_lpips, best_lpips,
               current_lr))

        row_train = [epoch, avg_loss, psnr_val_rgb, ssim_val_rgb, lpips_val_rgb]
        pd.DataFrame([row_train]).to_csv(train_log, mode="a", header=False, index=False)

    # 写 loss 日志（每个 epoch 都写）
    row_loss = [
        epoch,
        avg_loss,
        avg_loss_y,
        avg_loss_uv,
        avg_loss_yuv,
        avg_loss_percept,
        avg_loss_freq,
        avg_loss_color
    ]
    pd.DataFrame([row_loss]).to_csv(loss_log, mode="a", header=False, index=False)

    # scheduler step（保持你原逻辑：每epoch结束 step）
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]
    ))
    print("------------------------------------------------------------------")

    # 保存 latest（注意：必须包含 scheduler + best 信息，才能真正断点续训）
    torch.save({
        "epoch": epoch,
        "state_dict": model_restored.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_lpips": best_lpips,
        "best_epoch_psnr": best_epoch_psnr,
        "best_epoch_ssim": best_epoch_ssim,
        "best_epoch_lpips": best_epoch_lpips,
        "last_ssim_val": last_ssim_val,
        "last_lpips_val": last_lpips_val,
    }, os.path.join(model_dir, "model_latest.pth"))


total_finish_time = (time.time() - total_start_time)  # seconds
print("Total training time: {:.1f} hours".format((total_finish_time / 60 / 60)))
