#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import time
import zipfile
from pathlib import Path
from types import MethodType
from typing import Any, List, Mapping, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)

PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "train_inpainting"])
    ap.add_argument("--camera", type=str, default="camera_00", choices=["camera_00", "camera_02"])
    ap.add_argument("--result_root", type=str, required=True)
    ap.add_argument("--zip_path", type=str, required=True)

    ap.add_argument("--model_id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--process_res", type=int, default=756)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ckpt", type=str, default=None, help="Optional finetuned checkpoint path.")
    ap.add_argument("--req_h", type=int, default=3008)
    ap.add_argument("--req_w", type=int, default=4112)
    ap.add_argument("--use_autocast", action="store_true")
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "finetuned_nested", "finetuned_da3", "official_api"],
        help=(
            "auto: use finetuned_nested when ckpt is provided, otherwise official_api.\n"
            "finetuned_nested: match the normal finetune patched nested forward.\n"
            "finetuned_da3: alias of finetuned_nested.\n"
            "official_api: use official wrapper.inference, optionally after loading finetuned weights for A/B."
        ),
    )
    return ap.parse_args()


def parse_index_from_image_name(stem: str) -> int:
    match = re.search(r"(\d+)$", stem)
    if not match:
        raise ValueError(f"Image name '{stem}' has no numeric suffix.")
    return int(match.group(1))


def iter_rgb_files(cam_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(cam_dir.glob(ext))
    return sorted(files, key=lambda p: parse_index_from_image_name(p.stem))


def make_submission_zip(result_root: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for scene_dir in sorted([p for p in result_root.iterdir() if p.is_dir()]):
            for npy in sorted(scene_dir.glob("*.npy")):
                zf.write(npy, f"{scene_dir.name}/{npy.name}")
                written += 1
    if written == 0:
        raise RuntimeError(f"No predictions found under {result_root}")
    return written


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and len(obj) > 0 and all(isinstance(k, str) for k in obj.keys())


def _extract_state_dict(ckpt_obj: Any) -> Mapping[str, torch.Tensor]:
    if _looks_like_state_dict(ckpt_obj) and any(torch.is_tensor(v) for v in ckpt_obj.values()):
        return ckpt_obj
    if isinstance(ckpt_obj, Mapping):
        for key in ("model", "state_dict", "module", "net"):
            if key in ckpt_obj and _looks_like_state_dict(ckpt_obj[key]):
                return ckpt_obj[key]
    raise RuntimeError("Cannot find a valid state_dict in checkpoint.")


def _normalize_key_for_nested(key: str) -> str:
    prefixes = ("engine.module.", "module.", "model.", "wrapper.model.")
    out = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix):]
                changed = True
    return out


def load_finetuned_weights_strict_nested(wrapper: DepthAnything3, ckpt_path: Path):
    obj = torch.load(str(ckpt_path), map_location="cpu")
    raw_state = _extract_state_dict(obj)

    if not hasattr(wrapper, "model"):
        raise RuntimeError(f"Expected wrapper to have .model, got {type(wrapper)}")

    target_state = wrapper.model.state_dict()
    normalized = {}
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        norm_key = _normalize_key_for_nested(key)
        if norm_key in normalized:
            raise RuntimeError(f"Duplicate key after normalization: '{norm_key}'")
        normalized[norm_key] = value

    missing = sorted(set(target_state.keys()) - set(normalized.keys()))
    unexpected = sorted(set(normalized.keys()) - set(target_state.keys()))

    shape_mismatch = []
    for key in sorted(set(target_state.keys()) & set(normalized.keys())):
        if tuple(target_state[key].shape) != tuple(normalized[key].shape):
            shape_mismatch.append(
                f"{key}: ckpt{tuple(normalized[key].shape)} != model{tuple(target_state[key].shape)}"
            )

    if missing or unexpected or shape_mismatch:
        msg = [
            f"[CKPT] strict load failed: {ckpt_path}",
            f"  missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}",
        ]
        if missing:
            msg.append("  missing sample: " + ", ".join(missing[:10]))
        if unexpected:
            msg.append("  unexpected sample: " + ", ".join(unexpected[:10]))
        if shape_mismatch:
            msg.append("  shape mismatch sample: " + " | ".join(shape_mismatch[:5]))
        raise RuntimeError("\n".join(msg))

    wrapper.model.load_state_dict(normalized, strict=True)
    print(f"[CKPT] strict loaded into wrapper.model: {ckpt_path}")
    print(f"[CKPT] matched keys: {len(normalized)} / {len(target_state)}")


def resolve_ckpt_path(ckpt: Optional[str]) -> Optional[Path]:
    if not ckpt:
        return None

    ckpt_path = Path(ckpt).expanduser()
    if ckpt_path.exists():
        return ckpt_path

    fallback_map = {
        "best_16bit.pth": "best_model.pth",
        "last_16bit.pth": "last_model.pth",
        "best.pth": "best_model.pth",
        "last.pth": "last_model.pth",
    }
    alt_name = fallback_map.get(ckpt_path.name)
    if alt_name is not None:
        alt_path = ckpt_path.with_name(alt_name)
        if alt_path.exists():
            print(f"[CKPT] fallback from {ckpt_path.name} to {alt_path.name}")
            return alt_path

    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def resize_image_keep_long_side(img_rgb: np.ndarray, process_res: int) -> np.ndarray:
    h0, w0 = img_rgb.shape[:2]
    long_side = max(h0, w0)
    if long_side <= process_res:
        return img_rgb
    scale = float(process_res) / float(long_side)
    w1 = max(1, int(round(w0 * scale)))
    h1 = max(1, int(round(h0 * scale)))
    return cv2.resize(img_rgb, (w1, h1), interpolation=cv2.INTER_AREA)


def normalize_image_tensor(image_chw: torch.Tensor) -> torch.Tensor:
    mean = image_chw.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image_chw.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image_chw - mean) / std


def get_model_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
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
    return F.pad(x_bchw, (0, pad_w, 0, pad_h), mode="replicate"), pad_h, pad_w


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


def extract_depth_tensor(pred):
    if isinstance(pred, dict):
        depth = pred["depth"]
    elif hasattr(pred, "depth"):
        depth = pred.depth
    else:
        depth = pred

    if isinstance(depth, np.ndarray):
        tensor = torch.from_numpy(depth)
    elif torch.is_tensor(depth):
        tensor = depth
    else:
        raise TypeError(f"Unsupported prediction type: {type(depth)}")
    return squeeze_any_to_bhw(tensor.detach()).float().cpu()


def ensure_same_spatial_size(imgs: List[np.ndarray], batch_files: List[Path]):
    shapes = [(img.shape[0], img.shape[1]) for img in imgs]
    uniq = sorted(set(shapes))
    if len(uniq) != 1:
        raise RuntimeError(
            "Current low-level batched path requires same H/W inside one batch, "
            f"but got shapes={uniq} for files={[p.name for p in batch_files]}"
        )


def patch_nested_da3_for_inference(nested_model):
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
                    idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
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


def infer_finetuned_nested_batch(nested_model, imgs_bchw: torch.Tensor, use_autocast: bool) -> torch.Tensor:
    imgs = imgs_bchw[:, None, ...].contiguous()

    if use_autocast and imgs.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        class _NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        ctx = _NullCtx()

    with ctx:
        raw = nested_model(
            imgs,
            None,
            None,
            [],
            False,
            False,
            "saddle_balanced",
        )
    return extract_depth_tensor(raw)


def infer_official_api_single(wrapper: DepthAnything3, img_rgb: np.ndarray, process_res: int) -> np.ndarray:
    pred = wrapper.inference(
        image=[img_rgb],
        intrinsics=None,
        process_res=process_res,
        process_res_method="upper_bound_resize",
    )
    return extract_depth_tensor(pred)[0].numpy()


@torch.inference_mode()
def main():
    args = parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / args.split
    result_root = Path(args.result_root)
    zip_path = Path(args.zip_path)

    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    backend = args.backend
    if backend == "auto":
        backend = "finetuned_nested" if args.ckpt else "official_api"
    if backend == "finetuned_da3":
        backend = "finetuned_nested"

    ckpt_path = resolve_ckpt_path(args.ckpt)
    if backend == "finetuned_nested" and ckpt_path is None:
        raise ValueError(
            "backend=finetuned_nested requires --ckpt. "
            "Use --backend official_api if you want pretrained-only inference."
        )

    result_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 104)
    print(
        f"DA3 submission inference | split={args.split} | camera={args.camera} | "
        f"backend={backend}".center(104)
    )
    print("=" * 104)
    print(f"Split root:   {split_root}")
    print(f"Result root:  {result_root}")
    print(f"Model:        {args.model_id}")
    print(f"CKPT:         {str(ckpt_path) if ckpt_path else 'None (official pretrained)'}")
    print(f"Device:       {device}")
    print(f"Zip out:      {zip_path}")
    print(f"Process res:  {args.process_res}")
    print(f"Output size:  {args.req_w}x{args.req_h}")
    print(f"Batch size:   {args.batch}")
    print(f"Use autocast: {args.use_autocast}")
    print("Save dtype:   float16")
    print("-" * 104)

    t0 = time.time()
    wrapper = DepthAnything3.from_pretrained(args.model_id)
    if backend == "finetuned_nested":
        wrapper.model = patch_nested_da3_for_inference(wrapper.model)

    if ckpt_path is not None:
        load_finetuned_weights_strict_nested(wrapper, ckpt_path)

    wrapper = wrapper.to(device)
    wrapper.eval()
    wrapper.model.eval()

    print(f"[MODEL] wrapper.model type = {type(wrapper.model)}")
    print(f"[MODEL] model dtype        = {get_model_dtype(wrapper.model)}")
    print(f"[MODEL] finetune patch     = {'enabled' if backend == 'finetuned_nested' else 'disabled'}")
    print(f"Model ready in {time.time() - t0:.2f}s")

    scene_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])
    total = 0
    t_all = time.time()

    for si, scene_dir in enumerate(scene_dirs, start=1):
        scene = scene_dir.name
        cam_dir = scene_dir / args.camera
        if not cam_dir.exists():
            print(f"[{si}/{len(scene_dirs)}] Skip {scene}: missing {args.camera}")
            continue

        rgb_files = iter_rgb_files(cam_dir)
        if not rgb_files:
            print(f"[{si}/{len(scene_dirs)}] Skip {scene}: no images in {cam_dir}")
            continue

        out_scene = result_root / scene
        out_scene.mkdir(parents=True, exist_ok=True)
        print(f"[{si}/{len(scene_dirs)}] Scene={scene} | imgs={len(rgb_files)}")

        if backend == "official_api":
            for img_path in rgb_files:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                depth_m = infer_official_api_single(wrapper, img, args.process_res)
                idx = parse_index_from_image_name(img_path.stem)
                depth_cm = cv2.resize(depth_m.astype(np.float32) * 100.0, (args.req_w, args.req_h), interpolation=cv2.INTER_LINEAR)
                np.save(out_scene / f"{idx:04d}.npy", np.ascontiguousarray(depth_cm.astype(np.float16)))
                total += 1

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        elif backend == "finetuned_nested":
            for i in range(0, len(rgb_files), args.batch):
                batch_files = rgb_files[i:i + args.batch]
                imgs_np_small = []
                imgs_tensor_list = []

                for img_path in batch_files:
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError(f"Failed to read image: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img_small = resize_image_keep_long_side(img, args.process_res)
                    imgs_np_small.append(img_small)

                    img_tensor = torch.from_numpy(img_small.transpose(2, 0, 1).copy()).float().to(device) / 255.0
                    imgs_tensor_list.append(normalize_image_tensor(img_tensor))

                ensure_same_spatial_size(imgs_np_small, batch_files)

                imgs_bchw = torch.stack(imgs_tensor_list, dim=0)
                imgs_bchw = cast_to_model_dtype(imgs_bchw, wrapper.model)
                imgs_pad, pad_h, pad_w = pad_bchw_to_multiple(imgs_bchw, PATCH)

                pred_depth_m = infer_finetuned_nested_batch(wrapper.model, imgs_pad, args.use_autocast)
                pred_depth_m = crop_bhw_pad(pred_depth_m, pad_h, pad_w).numpy()

                for depth_m, img_path in zip(pred_depth_m, batch_files):
                    idx = parse_index_from_image_name(img_path.stem)
                    depth_cm = cv2.resize(depth_m.astype(np.float32) * 100.0, (args.req_w, args.req_h), interpolation=cv2.INTER_LINEAR)
                    np.save(out_scene / f"{idx:04d}.npy", np.ascontiguousarray(depth_cm.astype(np.float16)))
                    total += 1

                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"Unexpected backend: {backend}")

        print(f"    -> saved {len(list(out_scene.glob('*.npy')))} files to {out_scene}")

    written = make_submission_zip(result_root, zip_path)
    print("-" * 104)
    print(f"Zip created:  {zip_path}  (files: {written})")
    print(f"Done inference: scenes={len(scene_dirs)} images={total} time={time.time() - t_all:.1f}s")
    print("=" * 104)


if __name__ == "__main__":
    main()
