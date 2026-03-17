#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ctypes
import gc
import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional, List

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)

from dataloader import (
    BoosterTrainAuto,
    CenterCropOrPadToSize,
    Compose,
    RandomCropPadToSize,
    ResizeKeepAspectLowerBound,
)
from loss import AbsoluteMetricLoss, LossConfig

PATCH = 14

# =========================
# top-k save policy (固定策略，不加新参数)
# =========================
TOPK_KEEP = 4
TOPK_DIRNAME = "topk_ckpts"
TOPK_MANIFEST_NAME = "topk_manifest.json"
TRAINING_CKPT_DIRNAME = "training_ckpts"
BEST_EXPORT_NAME = "best_model.pth"
LAST_EXPORT_NAME = "last_model.pth"
SAVE_LAST_CHECKPOINT = True
PERIODIC_SAVE_START_EPOCH = 10
PERIODIC_SAVE_INTERVAL = 10
PERIODIC_EXPORT_DIRNAME = "periodic_exports"
PERIODIC_TRAINING_CKPT_DIRNAME = "periodic_training_ckpts"


# =========================
# dist / misc utils
# =========================
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_rank0() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_max_(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def dist_sum_int(x: int, device: torch.device) -> int:
    t = torch.tensor(float(x), device=device, dtype=torch.float64)
    all_reduce_sum_(t)
    return int(round(float(t.item())))


def dist_mean_scalar(x: torch.Tensor) -> float:
    t = x.detach().double().clone()
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return float(t.item())


def append_jsonl(path: Path, row: Dict[str, Any]):
    if not is_rank0():
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any):
    if not is_rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_ds_runtime_flags(ds_config_path: str):
    with open(ds_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    use_fp16 = bool(cfg.get("fp16", {}).get("enabled", False))
    use_bf16 = bool(cfg.get("bf16", {}).get("enabled", False))
    zero_stage = int(cfg.get("zero_optimization", {}).get("stage", 0) or 0)

    if use_fp16 and use_bf16:
        raise ValueError("ds_config.json cannot enable fp16 and bf16 at the same time.")

    return use_fp16, use_bf16, zero_stage


def maybe_trim_malloc():
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# =========================
# tensor / model utils
# =========================
def get_model_dtype(module: torch.nn.Module) -> torch.dtype:
    for p in module.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def cast_to_model_dtype(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    target_dtype = get_model_dtype(model)
    if x.dtype != target_dtype:
        x = x.to(dtype=target_dtype)
    return x


def pad_bchw_to_multiple(x_bchw: torch.Tensor, multiple: int = PATCH):
    _, _, hgt, wid = x_bchw.shape
    pad_h = (multiple - (hgt % multiple)) % multiple
    pad_w = (multiple - (wid % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x_bchw, 0, 0
    x = F.pad(x_bchw, (0, pad_w, 0, pad_h), mode="replicate")
    return x, pad_h, pad_w


def crop_bhw_pad(x_bhw: torch.Tensor, pad_h: int, pad_w: int):
    if pad_h:
        x_bhw = x_bhw[:, :-pad_h, :]
    if pad_w:
        x_bhw = x_bhw[:, :, :-pad_w]
    return x_bhw


def squeeze_any_to_bhw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 4 and t.shape[-1] == 1:
        t = t[..., 0]
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    while t.ndim > 3 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected tensor shape (want BHW): {tuple(t.shape)}")
    return t


def extract_attr_or_key(raw: Any, key: str):
    if isinstance(raw, dict):
        return raw[key]
    return getattr(raw, key)


def get_optional_attr_or_key(raw: Any, key: str):
    if isinstance(raw, dict):
        return raw.get(key, None)
    return getattr(raw, key, None)


def set_attr_or_key(raw: Any, key: str, value: Any):
    if isinstance(raw, dict):
        raw[key] = value
    else:
        setattr(raw, key, value)


def grad_global_norm_approx(model: torch.nn.Module) -> float:
    total = 0.0
    found = False
    for p in model.parameters():
        if p.grad is None:
            continue
        try:
            g = p.grad.detach().float()
        except Exception:
            continue
        found = True
        total += float((g * g).sum().item())
    if not found:
        return 0.0
    return total ** 0.5


# =========================
# patch official nested forward for training safety
# =========================
def patch_nested_da3_for_training(nested_model):
    """
    Patch official NestedDepthAnything3Net.forward() to avoid inplace ops
    during training, while still following the official total-forward path.
    """
    if not (hasattr(nested_model, "da3") and hasattr(nested_model, "da3_metric")):
        return nested_model

    def forward_safe(
        self,
        x: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ):
        if export_feat_layers is None:
            export_feat_layers = []

        output = self.da3(
            x,
            extrinsics,
            intrinsics,
            export_feat_layers=export_feat_layers,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        )
        metric_output = self.da3_metric(x)

        output_depth = extract_attr_or_key(output, "depth")
        output_depth_conf = get_optional_attr_or_key(output, "depth_conf")
        output_intrinsics = extract_attr_or_key(output, "intrinsics")
        output_extrinsics = get_optional_attr_or_key(output, "extrinsics")

        metric_depth = extract_attr_or_key(metric_output, "depth")
        metric_sky = get_optional_attr_or_key(metric_output, "sky")

        if output_depth_conf is None:
            output_depth_conf = torch.ones_like(output_depth)

        metric_depth = apply_metric_scaling(metric_depth, output_intrinsics)
        set_attr_or_key(metric_output, "depth", metric_depth)

        if metric_sky is not None:
            non_sky_mask = compute_sky_mask(metric_sky, threshold=0.3)
        else:
            non_sky_mask = torch.ones_like(output_depth, dtype=torch.bool)

        if int(non_sky_mask.sum().item()) > 10:
            depth_conf_ns = output_depth_conf[non_sky_mask]
            if depth_conf_ns.numel() > 0:
                depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
                median_conf = torch.quantile(depth_conf_sampled, 0.5)

                align_mask = compute_alignment_mask(
                    output_depth_conf,
                    non_sky_mask,
                    output_depth,
                    metric_depth,
                    median_conf,
                )

                if int(align_mask.sum().item()) > 10:
                    valid_depth = output_depth[align_mask]
                    valid_metric_depth = metric_depth[align_mask]
                    scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

                    if torch.isfinite(scale_factor):
                        output_depth = output_depth * scale_factor
                        set_attr_or_key(output, "depth", output_depth)

                        if output_extrinsics is not None:
                            extr = output_extrinsics.clone()
                            extr[:, :, :3, 3] = extr[:, :, :3, 3] * scale_factor
                            set_attr_or_key(output, "extrinsics", extr)

                        try:
                            set_attr_or_key(output, "is_metric", 1)
                        except Exception:
                            pass
                        try:
                            set_attr_or_key(output, "scale_factor", float(scale_factor.item()))
                        except Exception:
                            pass

        if metric_sky is not None:
            non_sky_mask = compute_sky_mask(metric_sky, threshold=0.3)
            non_sky_depth = output_depth[non_sky_mask]

            if non_sky_depth.numel() > 0:
                if non_sky_depth.numel() > 100000:
                    idx = torch.randint(
                        0,
                        non_sky_depth.numel(),
                        (100000,),
                        device=non_sky_depth.device,
                    )
                    sampled_depth = non_sky_depth[idx]
                else:
                    sampled_depth = non_sky_depth

                non_sky_max = torch.quantile(sampled_depth, 0.99)
                non_sky_max = min(float(non_sky_max.item()), 200.0)

                depth_fixed, depth_conf_fixed = set_sky_regions_to_max_depth(
                    output_depth,
                    output_depth_conf,
                    non_sky_mask,
                    max_depth=non_sky_max,
                )
                set_attr_or_key(output, "depth", depth_fixed)
                set_attr_or_key(output, "depth_conf", depth_conf_fixed)

        return output

    nested_model.forward = MethodType(forward_safe, nested_model)
    return nested_model


# =========================
# ZeRO-3 safe total forward
# =========================
def nested_forward_trainable(
    model_or_engine,
    imgs_bchw: torch.Tensor,
    K_b33: Optional[torch.Tensor] = None,
    use_autocast: bool = False,
):
    imgs = imgs_bchw[:, None, ...].contiguous()  # [B, 1, 3, H, W]
    ixts = K_b33[:, None, :, :].contiguous() if K_b33 is not None else None

    if use_autocast and imgs.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    with amp_ctx:
        raw = model_or_engine(
            imgs,
            None,
            ixts,
            [],
            False,
            False,
            "saddle_balanced",
        )

    depth_m = extract_attr_or_key(raw, "depth")
    depth_m = squeeze_any_to_bhw(depth_m)
    return depth_m


# =========================
# checkpoint save / top-k
# =========================
def save_zero3_16bit_to_path(engine, full_path: Path):
    """
    Save one single-file 16-bit model under ZeRO-3.
    """
    barrier()

    if is_rank0():
        full_path.parent.mkdir(parents=True, exist_ok=True)

    barrier()

    ok = engine.save_16bit_model(
        save_dir=str(full_path.parent),
        save_filename=full_path.name,
        exclude_frozen_parameters=False,
    )

    barrier()

    if is_rank0():
        if not ok:
            raise RuntimeError(
                "engine.save_16bit_model() returned False. "
                "Please set zero_optimization.stage3_gather_16bit_weights_on_model_save=true "
                "in ds_config.json"
            )
        print(f"  -> saved {full_path}", flush=True)

    barrier()


def atomic_torch_save(obj: Any, full_path: Path):
    tmp_path = full_path.with_name(f".{full_path.name}.tmp-rank{get_rank()}")
    torch.save(obj, str(tmp_path))
    os.replace(str(tmp_path), str(full_path))


def module_state_dict_to_cpu(module: torch.nn.Module):
    state_dict = module.state_dict()
    cpu_state_dict = OrderedDict()
    metadata = getattr(state_dict, "_metadata", None)

    for k, v in state_dict.items():
        cpu_state_dict[k] = v.detach().cpu() if torch.is_tensor(v) else v

    if metadata is not None:
        cpu_state_dict._metadata = metadata

    return cpu_state_dict


def save_rank0_state_dict_to_path(engine, full_path: Path):
    barrier()

    if is_rank0():
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state_dict = module_state_dict_to_cpu(engine.module)
        try:
            atomic_torch_save(cpu_state_dict, full_path)
            print(f"  -> exported {full_path}", flush=True)
        finally:
            del cpu_state_dict
            gc.collect()
            maybe_trim_malloc()

    barrier()


def save_model_export_to_path(engine, full_path: Path, zero_stage: int):
    if int(zero_stage) >= 3:
        save_zero3_16bit_to_path(engine, full_path)
    else:
        save_rank0_state_dict_to_path(engine, full_path)


def save_training_checkpoint(engine, save_root: Path, tag: str, client_state: Optional[Dict[str, Any]] = None):
    barrier()

    if is_rank0():
        save_root.mkdir(parents=True, exist_ok=True)

    barrier()

    engine.save_checkpoint(
        save_dir=str(save_root),
        tag=tag,
        client_state=client_state or {},
    )

    barrier()

    if is_rank0():
        print(f"  -> saved sharded checkpoint {save_root / tag}", flush=True)

    barrier()


def ckpt_rank_key(entry: Dict[str, Any]):
    """
    排序规则：
    1) val_rmse_cm 越小越好
    2) delta1.25 越大越好
    3) epoch 越靠后越好
    """
    return (
        float(entry["val_rmse_cm"]),
        -float(entry["delta1.25"]),
        -int(entry["epoch"]),
    )


def safe_unlink(path: Path):
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
    except FileNotFoundError:
        pass


def safe_remove_path(path: Path):
    if path.is_symlink() or path.is_file():
        safe_unlink(path)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def refresh_best_pointer(link_path: Path, target_path: Path):
    """
    best_16bit.pth 作为 top1 的软链接/硬链接/拷贝兜底
    """
    if not is_rank0():
        return

    safe_unlink(link_path)

    rel_target = os.path.relpath(str(target_path), str(link_path.parent))
    try:
        os.symlink(rel_target, str(link_path))
        return
    except Exception:
        pass

    try:
        os.link(str(target_path), str(link_path))
        return
    except Exception:
        pass

    shutil.copy2(str(target_path), str(link_path))


def write_topk_manifest(out_dir: Path, topk_entries: List[Dict[str, Any]]):
    manifest_path = out_dir / TOPK_MANIFEST_NAME
    obj = {
        "topk_keep": TOPK_KEEP,
        "entries": topk_entries,
    }
    write_json(manifest_path, obj)


def maybe_save_topk_checkpoint(
    engine,
    out_dir: Path,
    ep: int,
    val_stats: Dict[str, Any],
    topk_entries: List[Dict[str, Any]],
    zero_stage: int,
):
    """
    若当前 epoch 进入 top-k，则保存；否则不保存。
    返回更新后的 topk_entries。
    """
    ckpt_dir = out_dir / TOPK_DIRNAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    candidate = {
        "epoch": int(ep),
        "val_rmse_cm": float(val_stats["val_rmse_cm"]),
        "delta1.25": float(val_stats["delta1.25"]),
        "mae": float(val_stats["mae"]),
        "absrel": float(val_stats["absrel"]),
        "path": "",
        "export_path": "",
    }

    prev_best_epoch = int(topk_entries[0]["epoch"]) if topk_entries else None
    should_save = False
    if len(topk_entries) < TOPK_KEEP:
        should_save = True
    else:
        worst = sorted(topk_entries, key=ckpt_rank_key)[-1]
        if ckpt_rank_key(candidate) < ckpt_rank_key(worst):
            should_save = True

    if not should_save:
        return topk_entries

    filename = (
        f"ep{ep:03d}_"
        f"rmse{val_stats['val_rmse_cm']:.4f}_"
        f"d125{val_stats['delta1.25']:.2f}.pth"
    )
    tag = filename[:-4]
    training_root = out_dir / TRAINING_CKPT_DIRNAME
    training_path = training_root / tag

    save_training_checkpoint(
        engine,
        save_root=training_root,
        tag=tag,
        client_state={
            "epoch": int(ep),
            "val_stats": {k: float(v) if isinstance(v, np.floating) else v for k, v in val_stats.items()},
        },
    )

    candidate["path"] = str(training_path)
    topk_entries.append(candidate)
    topk_entries = sorted(topk_entries, key=ckpt_rank_key)

    while len(topk_entries) > TOPK_KEEP:
        removed = topk_entries.pop(-1)
        if is_rank0():
            safe_remove_path(Path(removed["path"]))

    if len(topk_entries) > 0:
        best_epoch = int(topk_entries[0]["epoch"])
        best_export = out_dir / BEST_EXPORT_NAME
        if best_epoch != prev_best_epoch:
            save_model_export_to_path(engine, best_export, zero_stage)

        if is_rank0():
            topk_entries[0]["export_path"] = str(best_export)
            refresh_best_pointer(out_dir / "best_16bit.pth", best_export)
            write_topk_manifest(out_dir, topk_entries)

    barrier()
    return topk_entries


def maybe_save_periodic_checkpoint(
    engine,
    out_dir: Path,
    ep: int,
    val_stats: Dict[str, Any],
    zero_stage: int,
):
    if ep < PERIODIC_SAVE_START_EPOCH:
        return
    if ep % PERIODIC_SAVE_INTERVAL != 0:
        return

    tag = f"ep{ep:03d}"
    save_training_checkpoint(
        engine,
        save_root=out_dir / PERIODIC_TRAINING_CKPT_DIRNAME,
        tag=tag,
        client_state={
            "epoch": int(ep),
            "kind": "periodic",
            "val_stats": {k: float(v) if isinstance(v, np.floating) else v for k, v in val_stats.items()},
        },
    )

    export_name = (
        f"{tag}_"
        f"rmse{val_stats['val_rmse_cm']:.4f}_"
        f"d125{val_stats['delta1.25']:.2f}.pth"
    )
    save_model_export_to_path(
        engine,
        out_dir / PERIODIC_EXPORT_DIRNAME / export_name,
        zero_stage,
    )


# =========================
# metrics / lr
# =========================
def get_lr_at_step(base_lr: float, global_step: int, total_steps: int, warmup_steps: int, poly_power: float) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and global_step < warmup_steps:
        alpha = float(global_step + 1) / float(max(warmup_steps, 1))
        return base_lr * alpha

    remain_steps = max(total_steps - warmup_steps, 1)
    passed = max(global_step - warmup_steps, 0)
    ratio = 1.0 - min(float(passed) / float(remain_steps), 1.0)
    return base_lr * (ratio ** poly_power)


def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def new_metric_accum(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "pix_cnt": torch.zeros((), device=device, dtype=torch.float64),
        "sse": torch.zeros((), device=device, dtype=torch.float64),
        "abs_err_sum": torch.zeros((), device=device, dtype=torch.float64),
        "absrel_sum": torch.zeros((), device=device, dtype=torch.float64),
        "d125_hit": torch.zeros((), device=device, dtype=torch.float64),
        "d120_hit": torch.zeros((), device=device, dtype=torch.float64),
        "d115_hit": torch.zeros((), device=device, dtype=torch.float64),
        "d110_hit": torch.zeros((), device=device, dtype=torch.float64),
        "d105_hit": torch.zeros((), device=device, dtype=torch.float64),
    }


def compute_metrics_exact_update(pred_cm: torch.Tensor, gt_cm: torch.Tensor, valid: torch.Tensor,
                                 accum: Dict[str, torch.Tensor]):
    pred_v = pred_cm[valid].float().clamp_min(1e-6)
    gt_v = gt_cm[valid].float().clamp_min(1e-6)

    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)

    accum["pix_cnt"] += float(pred_v.numel())
    accum["sse"] += ((pred_v - gt_v).double() ** 2).sum()
    accum["abs_err_sum"] += torch.abs(pred_v - gt_v).double().sum()
    accum["absrel_sum"] += (torch.abs(pred_v - gt_v) / gt_v).double().sum()

    accum["d125_hit"] += (ratio < 1.25).double().sum()
    accum["d120_hit"] += (ratio < 1.20).double().sum()
    accum["d115_hit"] += (ratio < 1.15).double().sum()
    accum["d110_hit"] += (ratio < 1.10).double().sum()
    accum["d105_hit"] += (ratio < 1.05).double().sum()


def scale_shift_align_batched(
    pred_bhw: torch.Tensor,
    gt_bhw: torch.Tensor,
    valid_bhw: torch.Tensor,
) -> torch.Tensor:
    aligned = pred_bhw.clone()
    bsz = pred_bhw.shape[0]

    for b in range(bsz):
        mask = valid_bhw[b]
        if int(mask.sum().item()) < 2:
            continue

        p = pred_bhw[b][mask].double()
        t = gt_bhw[b][mask].double()

        a00 = (p * p).sum()
        a01 = p.sum()
        a11 = torch.tensor(float(p.numel()), device=p.device, dtype=torch.float64)
        b0 = (p * t).sum()
        b1 = t.sum()

        det = a00 * a11 - a01 * a01
        if float(det.abs().item()) < 1e-12:
            scale = pred_bhw.new_tensor(1.0)
            shift = (t.mean() - p.mean()).to(pred_bhw.dtype)
        else:
            scale = ((a11 * b0 - a01 * b1) / det).to(pred_bhw.dtype)
            shift = ((-a01 * b0 + a00 * b1) / det).to(pred_bhw.dtype)

        aligned[b] = pred_bhw[b] * scale + shift

    return aligned


# =========================
# train / val
# =========================
def train_one_epoch(
    engine,
    loader,
    device,
    loss_fn: AbsoluteMetricLoss,
    max_depth_cm: Optional[float],
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    poly_power: float,
    use_k: bool,
    use_autocast: bool,
):
    engine.train()

    loss_sum = 0.0
    huber_sum = 0.0
    absrel_loss_sum = 0.0
    silog_sum = 0.0
    grad_loss_sum = 0.0

    loss_cnt = 0
    valid_ratio_sum = 0.0
    skipped_local = 0
    opt_steps_local = 0
    grad_norm_sum = 0.0
    grad_norm_cnt = 0
    last_lr = base_lr

    pbar = tqdm(loader, desc="train", leave=False, disable=not is_rank0())
    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["depth_cm"].to(device, non_blocking=True)
        vm = batch["valid"].to(device, non_blocking=True)
        K = batch["K"].to(device, non_blocking=True) if use_k else None

        if max_depth_cm is not None:
            gt = torch.clamp(gt, 1.0, max_depth_cm)

        img_pad, pad_h, pad_w = pad_bchw_to_multiple(img, PATCH)
        img_pad = cast_to_model_dtype(img_pad, engine.module)

        pred_m = nested_forward_trainable(
            engine,
            img_pad,
            K_b33=K,
            use_autocast=use_autocast,
        )

        pred_cm = crop_bhw_pad(pred_m, pad_h, pad_w) * 100.0

        if pred_cm.shape[-2:] != gt.shape[-2:]:
            pred_cm = F.interpolate(
                pred_cm[:, None],
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        valid = vm & torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred_cm)
        valid_ratio_sum += dist_mean_scalar(valid.float().mean())

        global_valid_count = dist_sum_int(int(valid.sum().item()), device)
        has_global_valid = global_valid_count > 0

        current_global_step = int(engine.global_steps)
        last_lr = get_lr_at_step(
            base_lr=base_lr,
            global_step=current_global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            poly_power=poly_power,
        )
        if engine.optimizer is not None:
            set_optimizer_lr(engine.optimizer, last_lr)

        if has_global_valid:
            loss_dict = loss_fn(pred_cm, gt, valid)
            loss = loss_dict["loss"]

            loss_sum += dist_mean_scalar(loss.detach())
            huber_sum += dist_mean_scalar(loss_dict["huber"])
            absrel_loss_sum += dist_mean_scalar(loss_dict["absrel"])
            silog_sum += dist_mean_scalar(loss_dict["silog"])
            grad_loss_sum += dist_mean_scalar(loss_dict["grad"])
            loss_cnt += 1
        else:
            loss = pred_cm.sum() * 0.0
            skipped_local += 1

        engine.backward(loss)

        grad_norm = grad_global_norm_approx(engine.module)
        if math.isfinite(grad_norm):
            grad_norm_sum += grad_norm
            grad_norm_cnt += 1

        prev_global_steps = int(engine.global_steps)
        engine.step()
        if int(engine.global_steps) > prev_global_steps:
            opt_steps_local += 1

        if is_rank0():
            postfix = {
                "loss": f"{(loss_sum / max(loss_cnt, 1)):.4f}",
                "skip": skipped_local,
                "opt": opt_steps_local,
                "lr": f"{last_lr:.2e}",
                "v": f"{(valid_ratio_sum / max(loss_cnt + skipped_local, 1)):.3f}",
                "gstep": int(engine.global_steps),
            }
            pbar.set_postfix(postfix)

    n_iter = loss_cnt + skipped_local
    global_steps_tensor = torch.tensor(float(int(engine.global_steps)), device=device, dtype=torch.float64)
    all_reduce_max_(global_steps_tensor)

    stats = {
        "train_loss": loss_sum / max(loss_cnt, 1),
        "train_huber": huber_sum / max(loss_cnt, 1),
        "train_absrel_loss": absrel_loss_sum / max(loss_cnt, 1),
        "train_silog": silog_sum / max(loss_cnt, 1),
        "train_grad_loss": grad_loss_sum / max(loss_cnt, 1),
        "skipped": dist_sum_int(skipped_local, device),
        "opt_steps": dist_sum_int(opt_steps_local, device),
        "global_steps": int(global_steps_tensor.item()),
        "valid_ratio": valid_ratio_sum / max(n_iter, 1),
        "grad_norm": grad_norm_sum / max(grad_norm_cnt, 1),
        "lr": last_lr,
        "num_iters": n_iter,
    }
    return stats


@torch.no_grad()
def validate(
    engine,
    loader,
    device,
    loss_fn: AbsoluteMetricLoss,
    max_depth_cm: Optional[float],
    use_k: bool,
    use_autocast: bool,
):
    engine.eval()

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    huber_sum = torch.zeros((), device=device, dtype=torch.float64)
    absrel_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    silog_sum = torch.zeros((), device=device, dtype=torch.float64)
    grad_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_cnt = torch.zeros((), device=device, dtype=torch.float64)
    skipped_local = torch.zeros((), device=device, dtype=torch.float64)
    valid_ratio_sum = 0.0
    n_batches = 0

    accum_final = new_metric_accum(device)
    accum_aligned = new_metric_accum(device)

    pbar = tqdm(loader, desc="val", leave=False, disable=not is_rank0())
    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["depth_cm"].to(device, non_blocking=True)
        vm = batch["valid"].to(device, non_blocking=True)
        K = batch["K"].to(device, non_blocking=True) if use_k else None

        if max_depth_cm is not None:
            gt = torch.clamp(gt, 1.0, max_depth_cm)

        img_pad, pad_h, pad_w = pad_bchw_to_multiple(img, PATCH)
        img_pad = cast_to_model_dtype(img_pad, engine.module)

        pred_m = nested_forward_trainable(
            engine,
            img_pad,
            K_b33=K,
            use_autocast=use_autocast,
        )

        pred_cm = crop_bhw_pad(pred_m, pad_h, pad_w) * 100.0

        if pred_cm.shape[-2:] != gt.shape[-2:]:
            pred_cm = F.interpolate(
                pred_cm[:, None],
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        valid = vm & torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred_cm)
        valid_ratio_sum += dist_mean_scalar(valid.float().mean())
        n_batches += 1

        global_valid_count = dist_sum_int(int(valid.sum().item()), device)
        has_global_valid = global_valid_count > 0
        if not has_global_valid:
            skipped_local += 1.0
            continue

        loss_dict = loss_fn(pred_cm, gt, valid)
        loss_sum += dist_mean_scalar(loss_dict["loss"])
        huber_sum += dist_mean_scalar(loss_dict["huber"])
        absrel_loss_sum += dist_mean_scalar(loss_dict["absrel"])
        silog_sum += dist_mean_scalar(loss_dict["silog"])
        grad_loss_sum += dist_mean_scalar(loss_dict["grad"])
        loss_cnt += 1.0

        if valid.any():
            compute_metrics_exact_update(pred_cm, gt, valid, accum_final)
            pred_aligned = scale_shift_align_batched(pred_cm, gt, valid)
            compute_metrics_exact_update(pred_aligned, gt, valid, accum_aligned)

        if is_rank0():
            final_rmse = math.sqrt(float(accum_final["sse"].item() / max(accum_final["pix_cnt"].item(), 1.0)))
            aligned_rmse = math.sqrt(float(accum_aligned["sse"].item() / max(accum_aligned["pix_cnt"].item(), 1.0)))
            pbar.set_postfix(
                loss=f"{float(loss_sum.item() / max(loss_cnt.item(), 1.0)):.4f}",
                rmse=f"{final_rmse:.3f}",
                rmse_aln=f"{aligned_rmse:.3f}",
            )

    for k in accum_final:
        all_reduce_sum_(accum_final[k])
        all_reduce_sum_(accum_aligned[k])

    all_reduce_sum_(loss_sum)
    all_reduce_sum_(huber_sum)
    all_reduce_sum_(absrel_loss_sum)
    all_reduce_sum_(silog_sum)
    all_reduce_sum_(grad_loss_sum)
    all_reduce_sum_(loss_cnt)
    all_reduce_sum_(skipped_local)

    pix_cnt_final = max(float(accum_final["pix_cnt"].item()), 1.0)
    pix_cnt_aligned = max(float(accum_aligned["pix_cnt"].item()), 1.0)

    final_rmse = math.sqrt(float(accum_final["sse"].item() / pix_cnt_final))
    aligned_rmse = math.sqrt(float(accum_aligned["sse"].item() / pix_cnt_aligned))

    stats = {
        "val_loss": float(loss_sum.item() / max(loss_cnt.item(), 1.0)),
        "val_huber": float(huber_sum.item() / max(loss_cnt.item(), 1.0)),
        "val_absrel_loss": float(absrel_loss_sum.item() / max(loss_cnt.item(), 1.0)),
        "val_silog": float(silog_sum.item() / max(loss_cnt.item(), 1.0)),
        "val_grad_loss": float(grad_loss_sum.item() / max(loss_cnt.item(), 1.0)),
        "val_rmse_cm": final_rmse,
        "val_aligned_rmse_cm": aligned_rmse,
        "skipped": int(round(float(skipped_local.item()))),
        "valid_ratio": valid_ratio_sum / max(n_batches, 1),
        "delta1.25": 100.0 * float(accum_final["d125_hit"].item() / pix_cnt_final),
        "delta1.20": 100.0 * float(accum_final["d120_hit"].item() / pix_cnt_final),
        "delta1.15": 100.0 * float(accum_final["d115_hit"].item() / pix_cnt_final),
        "delta1.10": 100.0 * float(accum_final["d110_hit"].item() / pix_cnt_final),
        "delta1.05": 100.0 * float(accum_final["d105_hit"].item() / pix_cnt_final),
        "mae": float(accum_final["abs_err_sum"].item() / pix_cnt_final),
        "absrel": float(accum_final["absrel_sum"].item() / pix_cnt_final),
        "rmse": final_rmse,
        "aligned_mae": float(accum_aligned["abs_err_sum"].item() / pix_cnt_aligned),
        "aligned_absrel": float(accum_aligned["absrel_sum"].item() / pix_cnt_aligned),
        "aligned_rmse": aligned_rmse,
    }
    return stats


# =========================
# print
# =========================
def print_epoch_summary(ep: int, total_epochs: int, train_stats: Dict[str, Any], val_stats: Dict[str, Any],
                        best_rmse: float, best_delta125: float, dt: float, topk_entries: List[Dict[str, Any]]):
    if not is_rank0():
        return

    print(
        f"[epoch {ep:03d}/{total_epochs:03d}] "
        f"lr={train_stats['lr']:.2e} "
        f"train_loss={train_stats['train_loss']:.4f} "
        f"skip={train_stats['skipped']} "
        f"opt={train_stats['opt_steps']} "
        f"gstep={train_stats['global_steps']} "
        f"time={dt:.1f}s",
        flush=True,
    )
    print(
        f"           "
        f"train(huber/absrel/silog/grad)="
        f"{train_stats['train_huber']:.4f}/"
        f"{train_stats['train_absrel_loss']:.4f}/"
        f"{train_stats['train_silog']:.4f}/"
        f"{train_stats['train_grad_loss']:.4f}",
        flush=True,
    )
    print(
        f"           "
        f"val_rmse_cm={val_stats['val_rmse_cm']:.3f} "
        f"val_aligned_rmse_cm={val_stats['val_aligned_rmse_cm']:.3f} "
        f"mae={val_stats['mae']:.3f} "
        f"absrel={val_stats['absrel']:.4f}",
        flush=True,
    )
    print(
        f"           "
        f"d1.25={val_stats['delta1.25']:.2f}% "
        f"d1.20={val_stats['delta1.20']:.2f}% "
        f"d1.15={val_stats['delta1.15']:.2f}% "
        f"d1.10={val_stats['delta1.10']:.2f}% "
        f"d1.05={val_stats['delta1.05']:.2f}%",
        flush=True,
    )
    print(
        f"           "
        f"valid_ratio(train/val)={train_stats['valid_ratio']:.4f}/{val_stats['valid_ratio']:.4f} "
        f"grad_norm≈{train_stats['grad_norm']:.4f}",
        flush=True,
    )
    print(
        f"           "
        f"best_rmse_cm={best_rmse:.3f} "
        f"best_d1.25={best_delta125:.2f}% "
        f"topk={[(e['epoch'], round(e['val_rmse_cm'], 3)) for e in topk_entries]}",
        flush=True,
    )


# =========================
# args
# =========================
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--train_txt", type=str, required=True)
    ap.add_argument("--val_txt", type=str, required=True)

    ap.add_argument("--camera", type=str, default="00", choices=["00", "02"])
    ap.add_argument("--baseline_unit", type=str, default="mm", choices=["mm", "cm", "m"])

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-7)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--crop_h", type=int, default=392)
    ap.add_argument("--crop_w", type=int, default=588)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_depth_cm", type=float, default=2000.0)

    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--poly_power", type=float, default=0.9)

    # kept for CLI compatibility; ignored in this official-total-forward version
    ap.add_argument("--aux_metric_loss_weight", type=float, default=0.0)
    ap.add_argument("--metric_scale_use_gt_k", action="store_true")

    ap.add_argument("--w_huber", type=float, default=1.0)
    ap.add_argument("--huber_delta_cm", type=float, default=20.0)
    ap.add_argument("--w_absrel", type=float, default=0.25)
    ap.add_argument("--w_silog", type=float, default=0.10)
    ap.add_argument("--silog_lambda", type=float, default=0.15)
    ap.add_argument("--w_grad", type=float, default=0.0)

    ap.add_argument("--ds_config", type=str, default="ds_config.json")
    ap.add_argument("--base_model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--local_rank", type=int, default=0)

    ap.add_argument("--use_k", action="store_true")
    ap.add_argument("--use_autocast", action="store_true", help="先不开更稳；跑通后再试混精")
    ap.add_argument("--detect_anomaly", action="store_true")
    return ap


# =========================
# main
# =========================
def main():
    args = build_argparser().parse_args()
    ds_use_fp16, ds_use_bf16, zero_stage = load_ds_runtime_flags(args.ds_config)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.crop_h % PATCH != 0 or args.crop_w % PATCH != 0:
        print(
            f"[WARN] crop_h={args.crop_h}, crop_w={args.crop_w} are not multiples of {PATCH}. "
            f"Recommended examples: 392x588, 518x714.",
            flush=True,
        )

    if args.aux_metric_loss_weight != 0 and is_rank0():
        print(
            "[WARN] In this ZeRO-3-safe official-total-forward version, "
            "aux_metric_loss_weight is ignored. Final depth only is supervised.",
            flush=True,
        )

    if args.metric_scale_use_gt_k and is_rank0():
        print(
            "[WARN] metric_scale_use_gt_k is ignored in this official-total-forward version.",
            flush=True,
        )

    if ds_use_fp16 and is_rank0():
        print("[WARN] fp16 is enabled in ds_config while this float32 training entry is being used.", flush=True)
    if ds_use_bf16 and is_rank0():
        print("[WARN] bf16 is enabled in ds_config while this float32 training entry is being used.", flush=True)

    set_seed(args.seed)

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    if is_rank0():
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] device={device} world_size={get_world_size()}", flush=True)
        print(f"[INFO] ds_config={args.ds_config}", flush=True)
        print(f"[INFO] out_dir={args.out_dir}", flush=True)
        print(f"[INFO] base_model={args.base_model}", flush=True)
        print(f"[INFO] train_txt={args.train_txt}", flush=True)
        print(f"[INFO] val_txt={args.val_txt}", flush=True)
        print(f"[INFO] lr={args.lr} weight_decay={args.weight_decay}", flush=True)
        print(f"[INFO] warmup_ratio={args.warmup_ratio} poly_power={args.poly_power}", flush=True)
        print(f"[INFO] crop_size={args.crop_h}x{args.crop_w}", flush=True)
        print(f"[INFO] use_k={args.use_k}", flush=True)
        print(f"[INFO] use_autocast={args.use_autocast}", flush=True)
        print(f"[INFO] zero_stage={zero_stage}", flush=True)
        print(
            f"[INFO] save policy: topk={TOPK_KEEP}, save_last={SAVE_LAST_CHECKPOINT}, "
            f"topk_dir={TOPK_DIRNAME}, training_ckpt_dir={TRAINING_CKPT_DIRNAME}, "
            f"best_export={BEST_EXPORT_NAME}, "
            f"periodic=every {PERIODIC_SAVE_INTERVAL} epochs (starting at epoch {PERIODIC_SAVE_START_EPOCH})",
            flush=True,
        )

    train_tf = Compose([
        ResizeKeepAspectLowerBound(args.crop_h, args.crop_w, multiple=PATCH),
        RandomCropPadToSize(args.crop_h, args.crop_w),
    ])
    val_tf = Compose([
        ResizeKeepAspectLowerBound(args.crop_h, args.crop_w, multiple=PATCH),
        CenterCropOrPadToSize(args.crop_h, args.crop_w),
    ])

    train_ds = BoosterTrainAuto(
        args.train_root,
        camera=args.camera,
        baseline_unit=args.baseline_unit,
        transform=train_tf,
        filelist_path=args.train_txt,
    )
    val_ds = BoosterTrainAuto(
        args.train_root,
        camera=args.camera,
        baseline_unit=args.baseline_unit,
        transform=val_tf,
        filelist_path=args.val_txt,
    )

    if is_rank0():
        print(f"[DATA] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=False,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        drop_last=False,
    )

    g = torch.Generator()
    g.manual_seed(args.seed + get_rank())

    common_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        **common_loader_kwargs,
    )

    if is_rank0():
        print("[MODEL] loading base model...", flush=True)

    wrapper = DepthAnything3.from_pretrained(args.base_model)
    wrapper.model = patch_nested_da3_for_training(wrapper.model)

    if not hasattr(wrapper, "model"):
        raise RuntimeError(f"Expected wrapper to have .model, got {type(wrapper)}")
    if not hasattr(wrapper.model, "da3") or not hasattr(wrapper.model, "da3_metric"):
        raise RuntimeError(
            f"Expected nested wrapper.model to have .da3 and .da3_metric, got {type(wrapper.model)}"
        )

    net = wrapper.model
    net.train()

    if is_rank0():
        print(f"[MODEL] wrapper.model type = {type(wrapper.model)}", flush=True)
        print(f"[MODEL] da3 type = {type(wrapper.model.da3)}", flush=True)
        print(f"[MODEL] da3_metric type = {type(wrapper.model.da3_metric)}", flush=True)
        n_total = sum(p.numel() for p in net.parameters())
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"[MODEL] trainable params = {n_trainable} / {n_total}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = AbsoluteMetricLoss(LossConfig(
        w_huber=args.w_huber,
        huber_delta_cm=args.huber_delta_cm,
        w_absrel=args.w_absrel,
        w_silog=args.w_silog,
        silog_lambda=args.silog_lambda,
        w_grad=args.w_grad,
    )).to(device)

    if is_rank0():
        print("[DS] initializing deepspeed...", flush=True)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=net,
        optimizer=optimizer,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        config=args.ds_config,
    )

    if is_rank0():
        print(f"[DTYPE] model dtype = {get_model_dtype(engine.module)}", flush=True)

    iters_per_epoch = len(train_loader)
    grad_accum = max(int(engine.gradient_accumulation_steps()), 1)
    total_opt_steps_est = max(int(math.ceil(args.epochs * iters_per_epoch / grad_accum)), 1)
    warmup_steps = int(total_opt_steps_est * args.warmup_ratio)

    if is_rank0():
        print(
            f"[SCHED] iters_per_epoch={iters_per_epoch} grad_accum={grad_accum} "
            f"total_opt_steps_est={total_opt_steps_est} warmup_steps={warmup_steps}",
            flush=True,
        )

    best_rmse = float("inf")
    best_delta125 = 0.0
    topk_entries: List[Dict[str, Any]] = []

    metrics_path = Path(args.out_dir) / "metrics.jsonl"
    if is_rank0() and metrics_path.exists():
        metrics_path.unlink()

    if is_rank0():
        topk_dir = Path(args.out_dir) / TOPK_DIRNAME
        topk_dir.mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / PERIODIC_EXPORT_DIRNAME).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / PERIODIC_TRAINING_CKPT_DIRNAME).mkdir(parents=True, exist_ok=True)
        write_topk_manifest(Path(args.out_dir), topk_entries)

    for ep in range(1, args.epochs + 1):
        train_sampler.set_epoch(ep)
        t0 = time.time()

        train_stats = train_one_epoch(
            engine=engine,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            max_depth_cm=args.max_depth_cm,
            base_lr=args.lr,
            total_steps=total_opt_steps_est,
            warmup_steps=warmup_steps,
            poly_power=args.poly_power,
            use_k=args.use_k,
            use_autocast=args.use_autocast,
        )

        val_stats = validate(
            engine=engine,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            max_depth_cm=args.max_depth_cm,
            use_k=args.use_k,
            use_autocast=args.use_autocast,
        )

        dt = time.time() - t0
        best_delta125 = max(best_delta125, val_stats["delta1.25"])

        if val_stats["val_rmse_cm"] < best_rmse:
            best_rmse = val_stats["val_rmse_cm"]

        topk_entries = maybe_save_topk_checkpoint(
            engine=engine,
            out_dir=Path(args.out_dir),
            ep=ep,
            val_stats=val_stats,
            topk_entries=topk_entries,
            zero_stage=zero_stage,
        )
        maybe_save_periodic_checkpoint(
            engine=engine,
            out_dir=Path(args.out_dir),
            ep=ep,
            val_stats=val_stats,
            zero_stage=zero_stage,
        )

        print_epoch_summary(
            ep,
            args.epochs,
            train_stats,
            val_stats,
            best_rmse if math.isfinite(best_rmse) else float("inf"),
            best_delta125,
            dt,
            topk_entries,
        )

        append_jsonl(metrics_path, {
            "epoch": ep,
            "time_sec": float(dt),
            **{k: float(v) if isinstance(v, np.floating) else v for k, v in train_stats.items()},
            **{k: float(v) if isinstance(v, np.floating) else v for k, v in val_stats.items()},
            "best_rmse": float(best_rmse),
            "best_delta125": float(best_delta125),
            "topk_epochs": [int(e["epoch"]) for e in topk_entries],
            "topk_rmse": [float(e["val_rmse_cm"]) for e in topk_entries],
        })

    if SAVE_LAST_CHECKPOINT:
        save_training_checkpoint(
            engine,
            save_root=Path(args.out_dir) / TRAINING_CKPT_DIRNAME,
            tag="last",
            client_state={"epoch": int(args.epochs), "kind": "last"},
        )
        last_path = Path(args.out_dir) / LAST_EXPORT_NAME
        save_model_export_to_path(engine, last_path, zero_stage)
        if is_rank0():
            refresh_best_pointer(Path(args.out_dir) / "last_16bit.pth", last_path)

    if is_rank0():
        # 再刷一次 manifest 和 best 指针，确保最终状态一致
        topk_entries = sorted(topk_entries, key=ckpt_rank_key)
        write_topk_manifest(Path(args.out_dir), topk_entries)
        if len(topk_entries) > 0:
            best_export = Path(args.out_dir) / BEST_EXPORT_NAME
            topk_entries[0]["export_path"] = str(best_export)
            refresh_best_pointer(Path(args.out_dir) / "best_16bit.pth", best_export)
            write_topk_manifest(Path(args.out_dir), topk_entries)
        print("Done.", flush=True)

    barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
