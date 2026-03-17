#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import os
import re
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14

# Keep OpenCV from spawning extra thread pools inside DataLoader workers.
cv2.setNumThreads(0)


# =========================================================
# util funcs
# =========================================================
def parse_numeric_suffix(stem: str) -> Optional[int]:
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return int(m.group(1))


def parse_dataset_txt(dataset_txt: str) -> Dict[str, List[str]]:
    """
    txt format:
      image_path gt_path calib_path

    where:
      image_path: relative to train_root, e.g. scene_xxx/camera_00/0001.png
      gt_path:    relative to train_root, e.g. scene_xxx/disp_00.npy
      calib_path: relative to train_root, e.g. scene_xxx/calib_00-02.xml
    """
    with open(dataset_txt, "r", encoding="utf-8") as f:
        gt_files = []
        image_files = []
        calib_files = []

        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            values = s.split()
            if len(values) != 3:
                raise ValueError(
                    f"Wrong format in dataset txt at line {ln}: {s}\n"
                    f"Expected 3 columns: image_path gt_path calib_path"
                )

            image_files.append(values[0].strip())
            gt_files.append(values[1].strip())
            calib_files.append(values[2].strip())

    if len(image_files) == 0:
        raise RuntimeError(f"No valid rows found in txt: {dataset_txt}")

    return {
        "image_paths": image_files,
        "gt_paths": gt_files,
        "calib_paths": calib_files,
    }


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def read_mask_binary_raw(path: Optional[str]) -> Optional[np.ndarray]:
    if (path is None) or (not os.path.exists(path)):
        return None
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m > 0


def read_mask_cat_raw(path: Optional[str]) -> Optional[np.ndarray]:
    if (path is None) or (not os.path.exists(path)):
        return None
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def read_fx_baseline_m(
    calib_xml: str,
    baseline_unit: str = "mm",
    camera: str = "00",
) -> Tuple[float, float]:
    fs = cv2.FileStorage(calib_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calib file: {calib_xml}")

    if camera == "00":
        P = fs.getNode("proj_matL").mat()
    elif camera == "02":
        P = fs.getNode("proj_matR").mat()
    else:
        fs.release()
        raise ValueError(f"camera must be '00' or '02', got {camera}")

    fx = float(P[0, 0])
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

    return fx, baseline_m


def disp_to_depth_cm(
    disp: np.ndarray,
    fx: float,
    baseline_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Only do physical conversion:
      disparity -> depth_m -> depth_cm
    No extra GT remapping.
    """
    disp = disp.astype(np.float32, copy=False)
    valid = disp > 0
    depth_m = np.zeros_like(disp, dtype=np.float32)
    depth_m[valid] = (fx * baseline_m) / disp[valid]
    depth_cm = depth_m * 100.0
    return depth_cm.astype(np.float32, copy=False), valid


def read_calib_K_and_size(calib_xml: str, camera: str) -> Tuple[np.ndarray, int, int]:
    fs = cv2.FileStorage(calib_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calib file: {calib_xml}")

    if camera == "00":
        P = fs.getNode("proj_matL").mat()
        w = int(fs.getNode("wL").real())
        h = int(fs.getNode("hL").real())
    elif camera == "02":
        P = fs.getNode("proj_matR").mat()
        w = int(fs.getNode("wR").real())
        h = int(fs.getNode("hR").real())
    else:
        fs.release()
        raise ValueError(f"camera must be '00' or '02', got {camera}")

    fs.release()
    if P is None or P.size == 0:
        raise ValueError(f"proj_mat not found in {calib_xml}")

    P = P.astype(np.float32)
    K = P[:, :3].copy().astype(np.float32)
    return K, w, h


def scale_K_numpy(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = K.copy().astype(np.float32)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def _resize_np(arr: np.ndarray, size_wh: Tuple[int, int], interpolation: int) -> np.ndarray:
    return cv2.resize(arr, size_wh, interpolation=interpolation)


def load_disp_npy_float32(path: str) -> np.ndarray:
    """
    Load disparity as a plain ndarray and close the backing mmap immediately.
    This avoids workers holding onto many mmap/file descriptors across epochs.
    """
    disp_raw = np.load(path, mmap_mode="r")
    try:
        disp = np.array(disp_raw, dtype=np.float32, copy=True)
    finally:
        mm = getattr(disp_raw, "_mmap", None)
        if mm is not None:
            mm.close()
        del disp_raw
    return disp


# =========================================================
# transform utils
# =========================================================
class Compose:
    def __init__(self, transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]]):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    """
    Close to PreRdw transform.Resize
    """

    def __init__(
        self,
        width: int,
        height: int,
        resize_target: bool = True,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resize_method: str = "lower_bound",
        image_interpolation_method: int = cv2.INTER_CUBIC,
    ):
        self._width = int(width)
        self._height = int(height)
        self._resize_target = bool(resize_target)
        self._keep_aspect_ratio = bool(keep_aspect_ratio)
        self._multiple_of = int(ensure_multiple_of)
        self._resize_method = str(resize_method)
        self._image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(
        self,
        x: float,
        min_val: int = 0,
        max_val: Optional[int] = None,
    ) -> int:
        y = int(np.round(x / self._multiple_of) * self._multiple_of)
        if max_val is not None and y > max_val:
            y = int(np.floor(x / self._multiple_of) * self._multiple_of)
        if y < min_val:
            y = int(np.ceil(x / self._multiple_of) * self._multiple_of)
        return y

    def get_size(self, width: int, height: int) -> Tuple[int, int]:
        scale_height = self._height / height
        scale_width = self._width / width

        if self._keep_aspect_ratio:
            if self._resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self._resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self._resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self._resize_method} not implemented")

        if self._resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height,
                min_val=self._height,
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width,
                min_val=self._width,
            )
        elif self._resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height,
                max_val=self._height,
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width,
                max_val=self._width,
            )
        elif self._resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self._resize_method} not implemented")

        return new_width, new_height

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample["image"]
        dep = sample["depth_cm"]
        val = sample["valid"]
        K = sample["K"]

        in_h, in_w = img.shape[:2]
        new_w, new_h = self.get_size(in_w, in_h)

        if new_w == in_w and new_h == in_h:
            sample["img_hw"] = (in_h, in_w)
            return sample

        img = _resize_np(img, (new_w, new_h), self._image_interpolation_method)

        if self._resize_target:
            dep = _resize_np(dep, (new_w, new_h), cv2.INTER_NEAREST)
            val = _resize_np(val.astype(np.float32), (new_w, new_h), cv2.INTER_NEAREST) > 0.5

            if "hard_mask" in sample:
                hard = sample["hard_mask"]
                sample["hard_mask"] = _resize_np(
                    hard.astype(np.uint8),
                    (new_w, new_h),
                    cv2.INTER_NEAREST,
                ) > 0

        sx = float(new_w) / float(in_w)
        sy = float(new_h) / float(in_h)
        K = K.copy().astype(np.float32)
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        sample["image"] = img
        sample["depth_cm"] = dep.astype(np.float32, copy=False)
        sample["valid"] = val.astype(np.bool_, copy=False)
        sample["K"] = K
        sample["img_hw"] = (new_h, new_w)
        return sample


class NormalizeImage:
    def __init__(self, mean, std):
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] = (sample["image"] - self._mean) / self._std
        return sample


class PrepareForNet:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "depth_cm" in sample:
            sample["depth_cm"] = np.ascontiguousarray(
                sample["depth_cm"].astype(np.float32, copy=False)
            )

        if "valid" in sample:
            sample["valid"] = np.ascontiguousarray(
                sample["valid"].astype(np.bool_, copy=False)
            )

        if "hard_mask" in sample:
            sample["hard_mask"] = np.ascontiguousarray(
                sample["hard_mask"].astype(np.bool_, copy=False)
            )

        if "K" in sample:
            sample["K"] = np.ascontiguousarray(
                sample["K"].astype(np.float32, copy=False)
            )

        return sample


# =========================================================
# compatibility wrappers expected by finetune.py
# =========================================================
class ResizeKeepAspectLowerBound:
    def __init__(self, target_h: int, target_w: int, multiple: int = PATCH):
        self.impl = Resize(
            width=int(target_w),
            height=int(target_h),
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=int(multiple),
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl(sample)


class RandomCropPadToSize:
    def __init__(self, crop_h: int, crop_w: int):
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample["image"]
        dep = sample["depth_cm"]
        val = sample["valid"]
        K = sample["K"].copy()

        H, W = img.shape[:2]
        ch, cw = self.crop_h, self.crop_w
        assert H >= ch and W >= cw, f"Wrong size: image {(H, W)} crop {(ch, cw)}"

        top = np.random.randint(0, H - ch + 1)
        left = np.random.randint(0, W - cw + 1)
        bottom = top + ch
        right = left + cw

        sample["image"] = img[top:bottom, left:right]
        sample["depth_cm"] = dep[top:bottom, left:right]
        sample["valid"] = val[top:bottom, left:right]

        if "hard_mask" in sample:
            sample["hard_mask"] = sample["hard_mask"][top:bottom, left:right]

        K[0, 2] -= float(left)
        K[1, 2] -= float(top)
        sample["K"] = K
        sample["img_hw"] = (ch, cw)
        return sample


class CenterCropOrPadToSize:
    def __init__(self, crop_h: int, crop_w: int):
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample["image"]
        dep = sample["depth_cm"]
        val = sample["valid"]
        K = sample["K"].copy()

        H, W = img.shape[:2]
        ch, cw = self.crop_h, self.crop_w

        if H < ch or W < cw:
            raise ValueError(
                f"CenterCropOrPadToSize expects resized image >= crop size, "
                f"got {(H, W)} vs {(ch, cw)}"
            )

        top = max(0, (H - ch) // 2)
        left = max(0, (W - cw) // 2)
        bottom = top + ch
        right = left + cw

        sample["image"] = img[top:bottom, left:right]
        sample["depth_cm"] = dep[top:bottom, left:right]
        sample["valid"] = val[top:bottom, left:right]

        if "hard_mask" in sample:
            sample["hard_mask"] = sample["hard_mask"][top:bottom, left:right]

        K[0, 2] -= float(left)
        K[1, 2] -= float(top)
        sample["K"] = K
        sample["img_hw"] = (ch, cw)
        return sample


# =========================================================
# txt-driven dataset
# =========================================================
@dataclass
class Item:
    scene: str
    index: int
    rgb_path: str
    disp_path: str
    calib_path: str
    mask_path: Optional[str] = None
    mask_cat_path: Optional[str] = None


class BoosterTrainAuto(Dataset):
    """
    txt format:
      image_path gt_path calib_path

    where gt_path is disparity npy path.
    """

    def __init__(
        self,
        data_root: str,
        camera: str = "00",
        baseline_unit: str = "mm",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        filelist_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.camera = camera
        self.baseline_unit = baseline_unit
        self.transform = transform
        self.filelist_path = filelist_path

        if self.filelist_path is None:
            raise ValueError("BoosterTrainAuto now expects filelist_path (txt) explicitly.")

        dataset_dict = parse_dataset_txt(self.filelist_path)

        self.items: List[Item] = []
        for img_rel, gt_rel, calib_rel in zip(
            dataset_dict["image_paths"],
            dataset_dict["gt_paths"],
            dataset_dict["calib_paths"],
        ):
            rgb_path = os.path.join(self.data_root, img_rel)
            disp_path = os.path.join(self.data_root, gt_rel)
            calib_path = os.path.join(self.data_root, calib_rel)

            if not os.path.exists(rgb_path):
                raise FileNotFoundError(f"Missing image file from txt: {rgb_path}")
            if not os.path.exists(disp_path):
                raise FileNotFoundError(f"Missing gt/disp file from txt: {disp_path}")
            if not os.path.exists(calib_path):
                raise FileNotFoundError(f"Missing calib file from txt: {calib_path}")

            scene_dir = os.path.dirname(calib_path)
            scene_name = os.path.basename(scene_dir)
            mask_path = os.path.join(scene_dir, f"mask_{self.camera}.png")

            mask_cat = os.path.join(scene_dir, "mask_cat.png")
            warped_mask_cat = os.path.join(scene_dir, "warped_mask_cat.png")
            mask_cat_path = (
                mask_cat if os.path.exists(mask_cat)
                else warped_mask_cat if os.path.exists(warped_mask_cat)
                else None
            )

            idx = parse_numeric_suffix(os.path.splitext(os.path.basename(rgb_path))[0])
            if idx is None:
                idx = len(self.items)

            self.items.append(
                Item(
                    scene=scene_name,
                    index=idx,
                    rgb_path=rgb_path,
                    disp_path=disp_path,
                    calib_path=calib_path,
                    mask_path=mask_path,
                    mask_cat_path=mask_cat_path,
                )
            )

        self.post_transform = Compose([
            NormalizeImage(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            PrepareForNet(),
        ])

        # 只缓存小型 calib 信息，避免 cache 越跑越大
        self._calib_cache: Dict[str, Dict[str, Any]] = {}

    def __len__(self):
        return len(self.items)

    def _load_calib_shared(self, item: Item) -> Dict[str, Any]:
        key = item.calib_path
        if key in self._calib_cache:
            return self._calib_cache[key]

        fx, baseline_m = read_fx_baseline_m(
            item.calib_path,
            baseline_unit=self.baseline_unit,
            camera=self.camera,
        )
        K_calib, w_cal, h_cal = read_calib_K_and_size(item.calib_path, self.camera)

        shared = {
            "fx": float(fx),
            "baseline_m": float(baseline_m),
            "K_calib": K_calib.astype(np.float32),
            "w_cal": int(w_cal),
            "h_cal": int(h_cal),
        }
        self._calib_cache[key] = shared
        return shared

    def __getitem__(self, idx: int):
        item = self.items[idx]

        image = read_rgb(item.rgb_path)
        H, W = image.shape[:2]
        shared = self._load_calib_shared(item)

        disp = load_disp_npy_float32(item.disp_path)
        depth_cm_raw, valid_disp_raw = disp_to_depth_cm(
            disp,
            shared["fx"],
            shared["baseline_m"],
        )
        del disp

        mask_valid_raw = read_mask_binary_raw(item.mask_path)
        hard_mask_raw = read_mask_cat_raw(item.mask_cat_path)

        if depth_cm_raw.shape != (H, W):
            depth_cm = _resize_np(depth_cm_raw, (W, H), cv2.INTER_NEAREST).astype(np.float32)
            valid_disp = _resize_np(
                valid_disp_raw.astype(np.float32),
                (W, H),
                cv2.INTER_NEAREST,
            ) > 0.5
        else:
            depth_cm = depth_cm_raw.astype(np.float32, copy=False)
            valid_disp = valid_disp_raw.astype(np.bool_, copy=False)

        if mask_valid_raw is None:
            mask_valid = np.ones((H, W), dtype=np.bool_)
        elif mask_valid_raw.shape != (H, W):
            mask_valid = _resize_np(
                mask_valid_raw.astype(np.float32),
                (W, H),
                cv2.INTER_NEAREST,
            ) > 0.5
        else:
            mask_valid = mask_valid_raw.astype(np.bool_, copy=False)

        hard_mask = None
        if hard_mask_raw is not None:
            if hard_mask_raw.shape != (H, W):
                hard_mask = _resize_np(
                    hard_mask_raw,
                    (W, H),
                    cv2.INTER_NEAREST,
                ) != 0
            else:
                hard_mask = hard_mask_raw != 0

        K_calib = shared["K_calib"]
        w_cal = shared["w_cal"]
        h_cal = shared["h_cal"]
        sx = float(W) / float(w_cal)
        sy = float(H) / float(h_cal)
        K_img = scale_K_numpy(K_calib, sx, sy)

        valid = mask_valid & valid_disp & np.isfinite(depth_cm) & (depth_cm > 0)

        sample: Dict[str, Any] = {
            "image": image,
            "depth_cm": depth_cm,
            "valid": valid,
            "K": K_img.astype(np.float32),
            "scene": item.scene,
            "index": item.index,
            "rgb_path": item.rgb_path,
            "img_hw": (H, W),
            "calib_hw": (h_cal, w_cal),
        }

        if hard_mask is not None:
            sample["hard_mask"] = hard_mask.astype(np.bool_, copy=False)

        if self.transform is not None:
            sample = self.transform(sample)

        sample = self.post_transform(sample)

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth_cm"] = torch.from_numpy(sample["depth_cm"])
        sample["valid"] = torch.from_numpy(sample["valid"])
        sample["K"] = torch.from_numpy(sample["K"])

        if "hard_mask" in sample:
            sample["hard_mask"] = torch.from_numpy(sample["hard_mask"])

        del image, depth_cm_raw, valid_disp_raw, mask_valid_raw, hard_mask_raw
        gc.collect()
        return sample


# =========================================================
# kept only for compatibility
# =========================================================
def build_scene_balanced_train_val_indices(
    items: List[Item],
    val_ratio: float = 0.1,
    seed: int = 42,
    val_max_per_scene: Optional[int] = 5,
) -> Tuple[List[int], List[int], List[str], List[str]]:
    if not items:
        return [], [], [], []

    scene_to_indices: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        scene_to_indices.setdefault(it.scene, []).append(i)

    scenes = sorted(scene_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(scenes)

    n_scenes = len(scenes)
    n_val_scenes = max(1, int(round(n_scenes * val_ratio)))
    n_val_scenes = min(n_val_scenes, max(1, n_scenes - 1)) if n_scenes > 1 else 1

    val_scenes = sorted(scenes[:n_val_scenes])
    train_scenes = sorted(scenes[n_val_scenes:])

    train_idx: List[int] = []
    val_idx: List[int] = []

    for s in train_scenes:
        idxs = sorted(scene_to_indices[s], key=lambda k: items[k].index)
        train_idx.extend(idxs)

    for s in val_scenes:
        idxs = sorted(scene_to_indices[s], key=lambda k: items[k].index)
        if (val_max_per_scene is None) or (val_max_per_scene <= 0) or (len(idxs) <= val_max_per_scene):
            chosen = idxs
        else:
            positions = np.linspace(0, len(idxs) - 1, num=val_max_per_scene)
            chosen = [idxs[int(round(p))] for p in positions]
            chosen = sorted(list(dict.fromkeys(chosen)))
        val_idx.extend(chosen)

    return train_idx, val_idx, train_scenes, val_scenes
