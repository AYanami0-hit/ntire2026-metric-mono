#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np


# 官方要求输出分辨率（仅用于 PNG 可视化，不影响点云重建）
REQ_H, REQ_W = 3008, 4112


def get_colormap(name: str):
    name = name.lower()
    cmap_dict = {
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "jet": cv2.COLORMAP_JET,
        "bone": cv2.COLORMAP_BONE,
        "gray": None,
    }
    if name not in cmap_dict:
        raise ValueError(
            f"Unsupported colormap: {name}. "
            f"Supported: {', '.join(cmap_dict.keys())}"
        )
    return cmap_dict[name]


def build_valid_mask(disp: np.ndarray, invalid_mode: str = "nonpositive") -> np.ndarray:
    finite_mask = np.isfinite(disp)
    if invalid_mode == "nonpositive":
        return finite_mask & (disp > 0)
    elif invalid_mode == "nan_only":
        return finite_mask
    else:
        raise ValueError("invalid_mode must be 'nonpositive' or 'nan_only'")


def normalize_disparity(
    disp: np.ndarray,
    robust: bool = True,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
    invalid_mode: str = "nonpositive",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将视差图归一化到 [0, 255]，只用有效区域统计。
    返回:
        disp_uint8: uint8 视差图
        valid_mask: 有效掩码
    """
    disp = disp.astype(np.float32)
    valid_mask = build_valid_mask(disp, invalid_mode=invalid_mode)

    if not np.any(valid_mask):
        raise ValueError("No valid disparity values found.")

    valid_disp = disp[valid_mask]

    if robust:
        vmin = np.percentile(valid_disp, min_percentile)
        vmax = np.percentile(valid_disp, max_percentile)
    else:
        vmin = np.min(valid_disp)
        vmax = np.max(valid_disp)

    if np.isclose(vmax, vmin):
        disp_uint8 = np.zeros_like(disp, dtype=np.uint8)
        return disp_uint8, valid_mask

    disp_clipped = np.clip(disp, vmin, vmax)
    disp_norm = (disp_clipped - vmin) / (vmax - vmin)
    disp_norm[~valid_mask] = 0.0
    disp_uint8 = (disp_norm * 255.0).round().astype(np.uint8)

    return disp_uint8, valid_mask


def colorize_disparity(
    disp: np.ndarray,
    colormap: str = "magma",
    robust: bool = True,
    invert: bool = False,
    invalid_mode: str = "nonpositive",
    invalid_black: bool = True,
) -> np.ndarray:
    disp_uint8, valid_mask = normalize_disparity(
        disp=disp,
        robust=robust,
        invalid_mode=invalid_mode,
    )

    if invert:
        disp_uint8 = 255 - disp_uint8

    cmap = get_colormap(colormap)
    if cmap is None:
        color = cv2.cvtColor(disp_uint8, cv2.COLOR_GRAY2BGR)
    else:
        color = cv2.applyColorMap(disp_uint8, cmap)

    if invalid_black:
        color[~valid_mask] = 0

    return color


def auto_find_xml(npy_path: Path) -> Path:
    """
    在 npy 同目录自动寻找 xml。
    优先规则：
    1. 只有一个 xml -> 直接用
    2. 多个 xml -> 优先包含 calib / camera / stereo / rect 的
    3. 再不行就按文件名排序取第一个
    """
    xml_files = sorted(npy_path.parent.glob("*.xml"))
    if len(xml_files) == 0:
        raise FileNotFoundError(f"No .xml calibration file found in: {npy_path.parent}")

    if len(xml_files) == 1:
        return xml_files[0]

    keywords = ["calib", "camera", "stereo", "rect"]
    preferred = []
    for x in xml_files:
        s = x.stem.lower()
        if any(k in s for k in keywords):
            preferred.append(x)

    if preferred:
        return sorted(preferred)[0]

    return xml_files[0]


def read_stereo_calibration(xml_path: Path):
    """
    从 OpenCV XML 中读取重建点云所需参数。
    优先使用 rectified projection matrix:
        proj_matL, proj_matR

    返回:
        fx, fy, cx, cy, baseline
    其中 baseline 单位与 XML 一致（你的这个大概率是 mm）
    """
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Failed to open calibration xml: {xml_path}")

    try:
        P1 = fs.getNode("proj_matL").mat()
        P2 = fs.getNode("proj_matR").mat()
        K1 = fs.getNode("mtxL").mat()
        baseline_node = fs.getNode("baselineLR")

        if P1 is not None and P1.size != 0:
            fx = float(P1[0, 0])
            fy = float(P1[1, 1])
            cx = float(P1[0, 2])
            cy = float(P1[1, 2])
        elif K1 is not None and K1.size != 0:
            fx = float(K1[0, 0])
            fy = float(K1[1, 1])
            cx = float(K1[0, 2])
            cy = float(K1[1, 2])
        else:
            raise ValueError(f"Cannot find proj_matL or mtxL in {xml_path}")

        baseline = None

        if P2 is not None and P2.size != 0:
            tx = float(P2[0, 3])
            fx_r = float(P2[0, 0])
            if abs(fx_r) > 1e-12 and abs(tx) > 1e-12:
                baseline = abs(tx / fx_r)

        if baseline is None or baseline <= 0:
            if not baseline_node.empty():
                baseline = float(baseline_node.real())

        if baseline is None or baseline <= 0:
            raise ValueError(f"Cannot infer baseline from {xml_path}")

        return fx, fy, cx, cy, baseline

    finally:
        fs.release()


def collect_npy_files(input_path: Path, recursive: bool = False) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise ValueError(f"Input file is not a .npy file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        pattern = "**/*.npy" if recursive else "*.npy"
        return sorted(input_path.glob(pattern))

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_output_paths(
    npy_path: Path,
    input_root: Path,
    output_root: Optional[Path],
) -> Tuple[Path, Path]:
    """
    返回:
        png_path, ply_path
    """
    if output_root is None:
        out_dir = npy_path.parent
    else:
        if input_root.is_file():
            rel_dir = Path(".")
        else:
            rel_dir = npy_path.parent.relative_to(input_root)
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{npy_path.stem}_disp.png"
    ply_path = out_dir / f"{npy_path.stem}_cloud.ply"
    return png_path, ply_path


def save_disparity_png(
    disp: np.ndarray,
    out_path: Path,
    colormap: str = "magma",
    robust: bool = True,
    invert: bool = False,
    invalid_mode: str = "nonpositive",
    resize_to_official: bool = False,
) -> np.ndarray:
    """
    保存伪彩色视差图，同时返回 color image(BGR) 给点云着色用。
    """
    disp_color = colorize_disparity(
        disp=disp,
        colormap=colormap,
        robust=robust,
        invert=invert,
        invalid_mode=invalid_mode,
        invalid_black=True,
    )

    save_img = disp_color
    if resize_to_official:
        save_img = cv2.resize(
            save_img,
            (REQ_W, REQ_H),
            interpolation=cv2.INTER_LINEAR,
        )

    ok = cv2.imwrite(str(out_path), save_img)
    if not ok:
        raise IOError(f"Failed to save PNG: {out_path}")

    return disp_color


def write_ply_binary_from_disparity(
    disp: np.ndarray,
    color_bgr: np.ndarray,
    ply_path: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    invalid_mode: str = "nonpositive",
    ply_step: int = 1,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
):
    """
    按 rectified stereo 模型:
        Z = fx * baseline / disparity
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    输出 binary little endian PLY，颜色来自视差渲染图。
    """
    if ply_step < 1:
        raise ValueError("ply_step must be >= 1")

    disp = disp.astype(np.float32)
    valid_mask = build_valid_mask(disp, invalid_mode=invalid_mode)

    # 采样
    disp_s = disp[::ply_step, ::ply_step]
    color_s = color_bgr[::ply_step, ::ply_step]
    valid_s = valid_mask[::ply_step, ::ply_step]

    # 深度
    z_s = np.zeros_like(disp_s, dtype=np.float32)
    z_s[valid_s] = (fx * baseline) / disp_s[valid_s]

    finite_xyz = np.isfinite(z_s) & valid_s
    if min_depth is not None:
        finite_xyz &= (z_s >= float(min_depth))
    if max_depth is not None:
        finite_xyz &= (z_s <= float(max_depth))

    vertex_count = int(finite_xyz.sum())
    if vertex_count == 0:
        raise ValueError("No valid 3D points to write into PLY.")

    # 预生成实际像素坐标（注意是原图坐标）
    u_coords = np.arange(0, disp.shape[1], ply_step, dtype=np.float32)
    v_coords = np.arange(0, disp.shape[0], ply_step, dtype=np.float32)

    ply_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))

        chunk_rows = 128
        h_s, w_s = disp_s.shape

        uu_full = u_coords[None, :]  # (1, W)

        for r0 in range(0, h_s, chunk_rows):
            r1 = min(r0 + chunk_rows, h_s)

            disp_chunk = disp_s[r0:r1]
            color_chunk = color_s[r0:r1]
            valid_chunk = finite_xyz[r0:r1]

            if not np.any(valid_chunk):
                continue

            vv = v_coords[r0:r1][:, None]  # (Hc, 1)

            z = np.zeros_like(disp_chunk, dtype=np.float32)
            z[valid_chunk] = (fx * baseline) / disp_chunk[valid_chunk]

            x = ((uu_full - cx) * z / fx).astype(np.float32)
            y = ((vv - cy) * z / fy).astype(np.float32)

            x_valid = x[valid_chunk]
            y_valid = y[valid_chunk]
            z_valid = z[valid_chunk]

            # BGR -> RGB
            rgb_valid = color_chunk[valid_chunk][:, ::-1]

            vertices = np.empty(len(x_valid), dtype=vertex_dtype)
            vertices["x"] = x_valid
            vertices["y"] = y_valid
            vertices["z"] = z_valid
            vertices["red"] = rgb_valid[:, 0]
            vertices["green"] = rgb_valid[:, 1]
            vertices["blue"] = rgb_valid[:, 2]

            vertices.tofile(f)


def process_one_file(
    npy_path: Path,
    input_root: Path,
    output_root: Optional[Path],
    xml_override: Optional[Path],
    colormap: str,
    robust: bool,
    invert: bool,
    resize_png: bool,
    invalid_mode: str,
    export_ply: bool,
    ply_step: int,
    min_depth: Optional[float],
    max_depth: Optional[float],
):
    disp = np.load(npy_path)
    if disp.ndim != 2:
        raise ValueError(f"Expected 2D disparity map, got shape {disp.shape} in {npy_path}")

    png_path, ply_path = build_output_paths(
        npy_path=npy_path,
        input_root=input_root,
        output_root=output_root,
    )

    # 1) 保存彩色视差图
    disp_color_bgr = save_disparity_png(
        disp=disp,
        out_path=png_path,
        colormap=colormap,
        robust=robust,
        invert=invert,
        invalid_mode=invalid_mode,
        resize_to_official=resize_png,
    )
    print(f"Saved PNG: {png_path}")

    # 2) 保存点云（默认开启）
    if export_ply:
        xml_path = xml_override if xml_override is not None else auto_find_xml(npy_path)
        fx, fy, cx, cy, baseline = read_stereo_calibration(xml_path)

        write_ply_binary_from_disparity(
            disp=disp,
            color_bgr=disp_color_bgr,
            ply_path=ply_path,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline=baseline,
            invalid_mode=invalid_mode,
            ply_step=ply_step,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        print(f"Saved PLY: {ply_path}")
        print(
            f"  [calib] xml={xml_path.name}, fx={fx:.6f}, fy={fy:.6f}, "
            f"cx={cx:.6f}, cy={cy:.6f}, baseline={baseline:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render disparity .npy to PNG and export PLY point cloud."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a .npy file or a directory containing .npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: save next to input file(s)",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Optional calibration xml path. If not set, auto-find *.xml in each npy folder",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search .npy files when input is a directory",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="magma",
        choices=["magma", "inferno", "plasma", "viridis", "turbo", "jet", "bone", "gray"],
        help="Colormap for disparity rendering. Default: magma",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert visualization",
    )
    parser.add_argument(
        "--no-robust",
        action="store_true",
        help="Disable robust percentile normalization for PNG rendering",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help=f"Resize PNG to official resolution: {REQ_H}x{REQ_W}",
    )
    parser.add_argument(
        "--invalid-mode",
        type=str,
        default="nonpositive",
        choices=["nonpositive", "nan_only"],
        help="How to judge invalid disparity values. Default: nonpositive",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Disable PLY export",
    )
    parser.add_argument(
        "--ply-step",
        type=int,
        default=1,
        help="Pixel sampling step for PLY export. 1=full resolution, 2/4 can reduce file size. Default: 1",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Optional min depth threshold for point cloud (same unit as baseline in xml)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional max depth threshold for point cloud (same unit as baseline in xml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_dir) if args.output_dir is not None else None
    xml_override = Path(args.xml) if args.xml is not None else None

    npy_files = collect_npy_files(input_path, recursive=args.recursive)
    if len(npy_files) == 0:
        print(f"No .npy files found in: {input_path}")
        return

    print(f"Found {len(npy_files)} npy file(s).")

    for npy_path in npy_files:
        try:
            process_one_file(
                npy_path=npy_path,
                input_root=input_path,
                output_root=output_root,
                xml_override=xml_override,
                colormap=args.colormap,
                robust=not args.no_robust,
                invert=args.invert,
                resize_png=args.resize,
                invalid_mode=args.invalid_mode,
                export_ply=not args.no_ply,
                ply_step=args.ply_step,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
        except Exception as e:
            print(f"[ERROR] {npy_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()