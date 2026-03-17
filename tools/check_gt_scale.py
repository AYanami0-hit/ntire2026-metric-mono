#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_fx_baseline_m(calib_xml: str, baseline_unit: str = "mm"):
    fs = cv2.FileStorage(calib_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calib file: {calib_xml}")

    P1 = fs.getNode("proj_matL").mat()
    if P1 is None or P1.size == 0:
        fs.release()
        raise RuntimeError(f"proj_matL missing in {calib_xml}")

    fx = float(P1[0, 0])
    baseline = float(fs.getNode("baselineLR").real())
    fs.release()

    if baseline_unit == "mm":
        baseline_m = baseline * 0.001
    elif baseline_unit == "cm":
        baseline_m = baseline * 0.01
    elif baseline_unit == "m":
        baseline_m = baseline
    else:
        raise ValueError("baseline_unit must be one of: mm, cm, m")

    return fx, baseline, baseline_m


def collect_scenes(train_root: Path):
    scenes = [p for p in train_root.iterdir() if p.is_dir()]
    scenes.sort(key=lambda p: p.name)
    return scenes


def first_image_in_camera(scene_dir: Path, camera: str):
    cam_dir = scene_dir / f"camera_{camera}"
    if not cam_dir.exists():
        return None
    files = []
    for ext in IMG_EXTS:
        files += list(cam_dir.glob(f"*{ext}"))
    if not files:
        return None
    files.sort()
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, required=True, help=".../NTIRE2026/train")
    ap.add_argument("--camera", type=str, default="00", choices=["00", "02"])
    ap.add_argument("--baseline_unit", type=str, default="mm", choices=["mm", "cm", "m"])
    ap.add_argument("--num_scenes", type=int, default=5, help="random sample scenes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_root = Path(args.train_root)
    assert train_root.exists(), f"not found: {train_root}"

    scenes = collect_scenes(train_root)
    if not scenes:
        raise RuntimeError(f"No scenes under {train_root}")

    rng = random.Random(args.seed)
    pick = scenes[:] if args.num_scenes >= len(scenes) else rng.sample(scenes, args.num_scenes)

    print("=" * 100)
    print(f"Check GT scale | train_root={train_root} | camera={args.camera} | baseline_unit={args.baseline_unit}")
    print("=" * 100)

    for scene_dir in pick:
        scene = scene_dir.name
        calib = scene_dir / "calib_00-02.xml"
        disp = scene_dir / f"disp_{args.camera}.npy"
        mask = scene_dir / f"mask_{args.camera}.png"
        img_path = first_image_in_camera(scene_dir, args.camera)

        if (not calib.exists()) or (not disp.exists()) or (img_path is None):
            print(f"[SKIP] {scene}: missing calib/disp/image")
            continue

        rgb = read_rgb(str(img_path))
        H, W = rgb.shape[:2]

        disp_np = np.load(disp).astype(np.float32)
        disp_hw = disp_np.shape

        if disp_np.shape != (H, W):
            disp_np_rs = cv2.resize(disp_np, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            disp_np_rs = disp_np

        fx, baseline_raw, baseline_m = read_fx_baseline_m(str(calib), baseline_unit=args.baseline_unit)

        valid_disp = disp_np_rs > 0
        if mask.exists():
            m = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
            if m is not None:
                if m.ndim == 3:
                    m = m[:, :, 0]
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                valid_disp = valid_disp & (m > 0)

        if valid_disp.sum() == 0:
            print(f"[WARN] {scene}: no valid disp pixels after mask")
            continue

        # disp -> depth
        depth_m = np.zeros_like(disp_np_rs, dtype=np.float32)
        depth_m[valid_disp] = (fx * baseline_m) / disp_np_rs[valid_disp]
        depth_cm = depth_m * 100.0

        # robust stats
        disp_med = float(np.median(disp_np_rs[valid_disp]))
        disp_p10 = float(np.percentile(disp_np_rs[valid_disp], 10))
        disp_p90 = float(np.percentile(disp_np_rs[valid_disp], 90))

        dep_cm_med = float(np.median(depth_cm[valid_disp]))
        dep_cm_p10 = float(np.percentile(depth_cm[valid_disp], 10))
        dep_cm_p90 = float(np.percentile(depth_cm[valid_disp], 90))

        print("-" * 100)
        print(f"[{scene}]")
        print(f"  img: {img_path.name}   img_hw={H}x{W}")
        print(f"  disp file: {disp.name} disp_hw={disp_hw}  (resized_to_img={disp_np_rs.shape != disp_hw})")
        print(f"  fx(px)={fx:.2f}   baselineLR(raw)={baseline_raw:.6f} ({args.baseline_unit})   baseline_m={baseline_m:.6f} m")
        print(f"  disp px   p10/med/p90 = {disp_p10:.3f} / {disp_med:.3f} / {disp_p90:.3f}")
        print(f"  depth cm  p10/med/p90 = {dep_cm_p10:.3f} / {dep_cm_med:.3f} / {dep_cm_p90:.3f}")

        # quick sanity hint
        # typical indoor: depth med maybe tens~hundreds cm; if you see 1e5cm or 1e-2cm it's likely unit wrong
        if dep_cm_med > 1e4:
            print("  [HINT] depth median > 1e4 cm (100m). baseline_unit 很可能错了（或 disp 单位不是像素）")
        if dep_cm_med < 1.0:
            print("  [HINT] depth median < 1 cm. baseline_unit 很可能错了（或 fx/baseline读错）")

    print("=" * 100)
    print("Done.")
    print("建议：分别用 baseline_unit=mm/cm/m 跑一遍，看哪个 depth(cm) 量级最合理。")


if __name__ == "__main__":
    main()
