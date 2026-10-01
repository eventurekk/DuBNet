import os
import random
import time
import argparse
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import losses
import numpy as np
import pandas as pd
import lpips
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset.data import get_training_data, get_validation_data
from models.KDDuBNet import Illum_YCRCB_Denoise_IN
from utils import network_parameters
import utils
from warmup_scheduler import GradualWarmupScheduler
from losses import SSIM_Loss


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="从 model_latest.pth 或指定 checkpoint 继续训练")
parser.add_argument("--resume_path", type=str, default="", help="指定继续训练的 checkpoint 路径")
parser.add_argument("--teacher_ckpt", type=str, default="/workspace/algorithm/END-main/model_bestPSNR.pth", help="0~1 教师模型 checkpoint")
args = parser.parse_args()


# ============================================================
# Seeds
# ============================================================
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


# ============================================================
# GPU
# ============================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Helpers
# ============================================================
def ycrcb_to_rgb_tensor(x: torch.Tensor) -> torch.Tensor:
    """YCrCb tensor -> RGB tensor. Input/output range: 0~1."""
    y = x[:, 0:1, :, :]
    cr = x[:, 1:2, :, :] - 0.5
    cb = x[:, 2:3, :, :] - 0.5

    r = y + 1.403 * cr
    g = y - 0.714 * cr - 0.344 * cb
    b = y + 1.773 * cb

    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, 0.0, 1.0)


def strip_module_prefix(sd: dict) -> dict:
    if not sd:
        return sd
    first_key = next(iter(sd.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def load_state_dict_from_ckpt(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return strip_module_prefix(sd)


def tensor_rgb_to_numpy01(x: torch.Tensor) -> np.ndarray:
    return x.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


# ============================================================
# Config / Paths
# ============================================================
with open("training.yaml", "r") as config:
    opt = yaml.safe_load(config)

Train = opt["TRAINING"]
OPT = opt["OPTIM"]
mode = opt["MODEL"]["MODE"]

model_dir = os.path.join(Train["SAVE_DIR"], mode, "models")
result_dir = os.path.join(Train["SAVE_DIR"], mode, "results")
log_dir = os.path.join(Train["SAVE_DIR"], mode, "log")
utils.mkdir(model_dir)
utils.mkdir(result_dir)
utils.mkdir(log_dir)

train_dir = Train["TRAIN_DIR"]
val_dir = Train["VAL_DIR"]


# ============================================================
# Distillation config
# ============================================================
TEACHER_CKPT = args.teacher_ckpt
TEACHER_FEATS = 128
STUDENT_FEATS = 64

LAMBDA_KD_OUT = 0.5
LAMBDA_KD_FEAT = 0.1
LAMBDA_KD_RGB = 0.2
KD_Y_W = 0.1
KD_UV_W = 0.1


# ============================================================
# Build teacher / student
# ============================================================
print("==> Build teacher and student")

teacher = Illum_YCRCB_Denoise_IN(n_feats=TEACHER_FEATS, return_feats=True).to(DEVICE)
teacher_sd = load_state_dict_from_ckpt(TEACHER_CKPT)
teacher.load_state_dict(teacher_sd, strict=True)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student = Illum_YCRCB_Denoise_IN(n_feats=STUDENT_FEATS, return_feats=True).to(DEVICE)
p_number = network_parameters(student)

kd_y_adapter = nn.Conv2d(STUDENT_FEATS, TEACHER_FEATS, kernel_size=1, bias=False).to(DEVICE)
kd_uv_adapter = nn.Conv2d(STUDENT_FEATS, TEACHER_FEATS, kernel_size=1, bias=False).to(DEVICE)


# ============================================================
# Loss functions
# ============================================================
char_criterion = losses.CharbonnierLoss()
ssim_criterion = losses.SSIM_Loss()
percept_criterion = losses.PerceptualLoss().to(DEVICE)
freq_criterion = losses.FocalFrequencyLoss().to(DEVICE)
color_criterion = losses.ColorLoss().to(DEVICE)
val_ssim_criterion = SSIM_Loss().to(DEVICE)
val_lpips_criterion = lpips.LPIPS(net="alex").to(DEVICE)


# ============================================================
# Optimizer / Scheduler
# ============================================================
optimizer = optim.Adam(
    list(student.parameters()) + list(kd_y_adapter.parameters()) + list(kd_uv_adapter.parameters()),
    lr=float(OPT["LR_INITIAL"]),
    betas=(0.9, 0.999),
    eps=1e-8,
)

warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    OPT["EPOCHS"] - warmup_epochs,
    eta_min=float(OPT["LR_MIN"]),
)
scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=1,
    total_epoch=warmup_epochs,
    after_scheduler=scheduler_cosine,
)


# ============================================================
# DataLoaders
# ============================================================
print("==> Loading datasets")
train_dataset = get_training_data(train_dir, {"patch_size": Train["TRAIN_PS"]})
train_loader = DataLoader(
    train_dataset,
    batch_size=OPT["BATCH"],
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=False,
)

val_dataset = get_validation_data(val_dir, {"patch_size": Train["VAL_PS"]})
VAL_BS = 8
val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BS,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
)


# ============================================================
# Logs
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_log = os.path.join(log_dir, f"kd_train_{timestamp}.csv")
loss_log = os.path.join(log_dir, f"kd_loss_{timestamp}.csv")

pd.DataFrame(columns=["epoch", "train_loss", "val_psnr", "val_ssim", "val_lpips", "lr"]).to_csv(train_log, index=False)
pd.DataFrame(columns=[
    "epoch", "train_loss",
    "loss_y", "loss_uv", "loss_yuv",
    "loss_percept", "loss_freq", "loss_color",
    "loss_kd_out", "loss_kd_feat", "loss_kd_rgb"
]).to_csv(loss_log, index=False)


# ============================================================
# Resume
# ============================================================
start_epoch = 1
best_psnr = 0.0
best_ssim = 0.0
best_lpips = 1e9
best_epoch_psnr = 0
best_epoch_ssim = 0
best_epoch_lpips = 0
last_ssim_val = 0.0
last_lpips_val = 1.0

if args.resume:
    resume_path = args.resume_path.strip()
    if resume_path == "":
        resume_path = os.path.join(model_dir, "model_latest.pth")
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"[Resume] 找不到 checkpoint: {resume_path}")

    print(f"==> [Resume] Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=DEVICE)

    student.load_state_dict(ckpt["state_dict"], strict=True)
    if "kd_y_adapter" in ckpt:
        kd_y_adapter.load_state_dict(ckpt["kd_y_adapter"], strict=True)
    if "kd_uv_adapter" in ckpt:
        kd_uv_adapter.load_state_dict(ckpt["kd_uv_adapter"], strict=True)

    optimizer.load_state_dict(ckpt["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(DEVICE, non_blocking=True)

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    ckpt_epoch = int(ckpt.get("epoch", 0))
    start_epoch = ckpt_epoch + 1
    best_psnr = float(ckpt.get("best_psnr", 0.0))
    best_ssim = float(ckpt.get("best_ssim", 0.0))
    best_lpips = float(ckpt.get("best_lpips", 1e9))
    best_epoch_psnr = int(ckpt.get("best_epoch_psnr", 0))
    best_epoch_ssim = int(ckpt.get("best_epoch_ssim", 0))
    best_epoch_lpips = int(ckpt.get("best_epoch_lpips", 0))
    last_ssim_val = float(ckpt.get("last_ssim_val", 0.0))
    last_lpips_val = float(ckpt.get("last_lpips_val", 1.0))

    print(f"==> [Resume] start_epoch={start_epoch}, best_psnr={best_psnr:.4f}, best_ssim={best_ssim:.4f}, best_lpips={best_lpips:.4f}")
else:
    print("==> [Fresh] Start distillation training from scratch")
    scheduler.step()


# ============================================================
# Summary
# ============================================================
print(f"""==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patch size:   {Train['TRAIN_PS']}x{Train['TRAIN_PS']}
    Val patch size:     {Train['VAL_PS']}x{Train['VAL_PS']}
    Teacher ckpt:       {TEACHER_CKPT}
    Teacher/Student:    {TEACHER_FEATS}/{STUDENT_FEATS}
    Student params:     {p_number}
    Start/End epochs:   {start_epoch}~{OPT['EPOCHS']}
    Batch size:         {OPT['BATCH']}
    Val batch size:     {VAL_BS}
    LR initial:         {OPT['LR_INITIAL']}
    Train log:          {train_log}
    Loss log:           {loss_log}
------------------------------------------------------------------""")


# ============================================================
# Validation function
# ============================================================
def validate(epoch: int):
    student.eval()
    psnr_list = []
    ssim_list = []
    lpips_list = []
    target_size = (256, 256)

    with torch.no_grad():
        for data_val in val_loader:
            target = data_val[0].to(DEVICE, non_blocking=True)
            input_ = data_val[1].to(DEVICE, non_blocking=True)
            _, _, restored, _, _ = student(input_)

            restored_rgb = ycrcb_to_rgb_tensor(restored)
            target_rgb = ycrcb_to_rgb_tensor(target)

            # SSIM: tensor 0~1
            ssim_loss = val_ssim_criterion(restored_rgb, target_rgb)
            ssim_list.append(1.0 - float(ssim_loss.item()))

            restored_np = tensor_rgb_to_numpy01(restored_rgb)
            target_np = tensor_rgb_to_numpy01(target_rgb)

            for b in range(restored_np.shape[0]):
                # PSNR: utils.PSNR 默认按 0~255 使用，这里乘回 255
                psnr_list.append(utils.PSNR(target_np[b] * 255.0, restored_np[b] * 255.0))

                # LPIPS: input [-1,1], resize 256x256
                r_resized = cv2.resize(restored_np[b], target_size, interpolation=cv2.INTER_CUBIC)
                t_resized = cv2.resize(target_np[b], target_size, interpolation=cv2.INTER_CUBIC)
                r_tensor = torch.from_numpy(r_resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float() * 2.0 - 1.0
                t_tensor = torch.from_numpy(t_resized).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float() * 2.0 - 1.0
                lpips_score = float(val_lpips_criterion(r_tensor, t_tensor).item())
                lpips_list.append(float(np.clip(lpips_score, 0.0, 1.0)))

    psnr_val = float(np.mean(psnr_list)) if psnr_list else 0.0
    ssim_val = float(np.mean(ssim_list)) if ssim_list else 0.0
    lpips_val = float(np.mean(lpips_list)) if lpips_list else 1.0
    return psnr_val, ssim_val, lpips_val


# ============================================================
# Training loop
# ============================================================
total_start = time.time()

for epoch in range(start_epoch, OPT["EPOCHS"] + 1):
    epoch_start = time.time()
    student.train()
    kd_y_adapter.train()
    kd_uv_adapter.train()

    epoch_loss = 0.0
    sums = {
        "loss_y": 0.0,
        "loss_uv": 0.0,
        "loss_yuv": 0.0,
        "loss_percept": 0.0,
        "loss_freq": 0.0,
        "loss_color": 0.0,
        "loss_kd_out": 0.0,
        "loss_kd_feat": 0.0,
        "loss_kd_rgb": 0.0,
    }
    batch_count = 0

    for data in tqdm(train_loader):
        target = data[0].to(DEVICE, non_blocking=True)
        input_ = data[1].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_y, t_uv, t_out, t_yf, t_uvf = teacher(input_)

        s_y, s_uv, s_out, s_yf, s_uvf = student(input_)

        target_y = target[:, 0:1, :, :]
        target_uv = target[:, 1:3, :, :]

        loss_y = char_criterion(s_y, target_y) + ssim_criterion(s_y, target_y)
        loss_uv = char_criterion(s_uv, target_uv) + ssim_criterion(s_uv, target_uv)
        loss_yuv = char_criterion(s_out, target) + ssim_criterion(s_out, target)

        s_rgb = ycrcb_to_rgb_tensor(s_out)
        t_rgb = ycrcb_to_rgb_tensor(t_out)
        s_rgb_m11 = s_rgb * 2.0 - 1.0
        t_rgb_m11 = t_rgb * 2.0 - 1.0

        loss_percept = percept_criterion(s_rgb_m11, t_rgb_m11)
        loss_color = color_criterion(s_rgb_m11, t_rgb_m11)
        loss_freq = freq_criterion(s_rgb, t_rgb)

        loss_sup = (
            0.01 * loss_y
            + 0.1 * loss_uv
            + loss_yuv
            + 0.08 * loss_percept
            + 0.04 * loss_freq
            + 0.03 * loss_color
        )

        kd_out = char_criterion(s_out, t_out)
        kd_y = char_criterion(s_y, t_y)
        kd_uv = char_criterion(s_uv, t_uv)
        loss_kd_out = kd_out + KD_Y_W * kd_y + KD_UV_W * kd_uv
        loss_kd_rgb = F.l1_loss(s_rgb, t_rgb)
        loss_kd_feat = F.l1_loss(kd_y_adapter(s_yf), t_yf) + F.l1_loss(kd_uv_adapter(s_uvf), t_uvf)

        loss = loss_sup + LAMBDA_KD_OUT * loss_kd_out + LAMBDA_KD_FEAT * loss_kd_feat + LAMBDA_KD_RGB * loss_kd_rgb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(kd_y_adapter.parameters()) + list(kd_uv_adapter.parameters()), max_norm=3.0)
        optimizer.step()

        epoch_loss += float(loss.item())
        sums["loss_y"] += float(loss_y.item())
        sums["loss_uv"] += float(loss_uv.item())
        sums["loss_yuv"] += float(loss_yuv.item())
        sums["loss_percept"] += float(loss_percept.item())
        sums["loss_freq"] += float(loss_freq.item())
        sums["loss_color"] += float(loss_color.item())
        sums["loss_kd_out"] += float(loss_kd_out.item())
        sums["loss_kd_feat"] += float(loss_kd_feat.item())
        sums["loss_kd_rgb"] += float(loss_kd_rgb.item())
        batch_count += 1

    denom = max(batch_count, 1)
    avg_loss = epoch_loss / denom
    avg = {k: v / denom for k, v in sums.items()}

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # default values if this epoch does not validate
    psnr_val = 0.0
    ssim_val = last_ssim_val
    lpips_val = last_lpips_val

    if epoch % Train["VAL_AFTER_EVERY"] == 0:
        psnr_val, ssim_val, lpips_val = validate(epoch)
        last_ssim_val = ssim_val
        last_lpips_val = lpips_val

        state = {
            "epoch": epoch,
            "state_dict": student.state_dict(),
            "kd_y_adapter": kd_y_adapter.state_dict(),
            "kd_uv_adapter": kd_uv_adapter.state_dict(),
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
            "student_feats": STUDENT_FEATS,
            "teacher_feats": TEACHER_FEATS,
            "teacher_ckpt": TEACHER_CKPT,
        }

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_epoch_psnr = epoch
            state["best_psnr"] = best_psnr
            state["best_epoch_psnr"] = best_epoch_psnr
            torch.save(state, os.path.join(model_dir, "model_bestPSNR.pth"))

        if ssim_val > best_ssim:
            best_ssim = ssim_val
            best_epoch_ssim = epoch
            state["best_ssim"] = best_ssim
            state["best_epoch_ssim"] = best_epoch_ssim
            torch.save(state, os.path.join(model_dir, "model_bestSSIM.pth"))

        if lpips_val < best_lpips:
            best_lpips = lpips_val
            best_epoch_lpips = epoch
            state["best_lpips"] = best_lpips
            state["best_epoch_lpips"] = best_epoch_lpips
            torch.save(state, os.path.join(model_dir, "model_bestLPIPS.pth"))

        print(
            "[epoch %d PSNR: %.4f SSIM: %.4f LPIPS: %.4f | "
            "best_psnr(epoch %d): %.4f | best_ssim(epoch %d): %.4f | best_lpips(epoch %d): %.4f | LR: %.6f]"
            % (
                epoch, psnr_val, ssim_val, lpips_val,
                best_epoch_psnr, best_psnr,
                best_epoch_ssim, best_ssim,
                best_epoch_lpips, best_lpips,
                current_lr,
            )
        )

        pd.DataFrame([[epoch, avg_loss, psnr_val, ssim_val, lpips_val, current_lr]]).to_csv(train_log, mode="a", header=False, index=False)

    pd.DataFrame([[
        epoch,
        avg_loss,
        avg["loss_y"],
        avg["loss_uv"],
        avg["loss_yuv"],
        avg["loss_percept"],
        avg["loss_freq"],
        avg["loss_color"],
        avg["loss_kd_out"],
        avg["loss_kd_feat"],
        avg["loss_kd_rgb"],
    ]]).to_csv(loss_log, mode="a", header=False, index=False)

    latest_state = {
        "epoch": epoch,
        "state_dict": student.state_dict(),
        "kd_y_adapter": kd_y_adapter.state_dict(),
        "kd_uv_adapter": kd_uv_adapter.state_dict(),
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
        "student_feats": STUDENT_FEATS,
        "teacher_feats": TEACHER_FEATS,
        "teacher_ckpt": TEACHER_CKPT,
    }
    torch.save(latest_state, os.path.join(model_dir, "model_latest.pth"))

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start, epoch_loss, current_lr))
    print("------------------------------------------------------------------")

print("Total training time: {:.1f} hours".format((time.time() - total_start) / 3600.0))
