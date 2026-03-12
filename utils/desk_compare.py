import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


def normalize_image_name(name: str) -> str:
    value = Path(str(name)).name
    return Path(value).stem


def load_split_manifest(path: str):
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _split_names(payload: Dict[str, Any], split: str) -> List[str]:
    candidates = [
        payload.get(f"{split}_image_names"),
        payload.get(f"{split}_images"),
        payload.get(split),
        payload.get("splits", {}).get(split) if isinstance(payload.get("splits"), dict) else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            return [normalize_image_name(item) for item in candidate]
    return []


def partition_cameras(cam_infos, manifest: Optional[Dict[str, Any]], eval_enabled: bool, default_test_names: List[str]):
    if manifest is None:
        test_names = {normalize_image_name(name) for name in default_test_names}
        train = [cam for cam in cam_infos if not test_names or normalize_image_name(cam.image_name) not in test_names]
        test = [cam for cam in cam_infos if normalize_image_name(cam.image_name) in test_names]
        if not eval_enabled:
            return cam_infos, []
        return train, test

    train_names = set(_split_names(manifest, "train"))
    test_names = set(_split_names(manifest, "test"))
    if not train_names and not test_names:
        raise ValueError("Split manifest does not define train/test image lists.")

    train = []
    test = []
    for cam in cam_infos:
        normalized = normalize_image_name(cam.image_name)
        if normalized in test_names:
            test.append(cam)
        elif normalized in train_names:
            train.append(cam)
    if not train:
        raise ValueError("No train cameras matched the split manifest.")
    if eval_enabled and not test:
        raise ValueError("No test cameras matched the split manifest.")
    if not eval_enabled:
        return train, []
    return train, test


def resolve_monitor_target(manifest: Optional[Dict[str, Any]], split: str, requested_index: int, requested_view_name: str, cameras):
    camera_list = list(cameras)
    if not camera_list:
        raise ValueError(f"Cannot resolve monitor view from empty `{split}` split.")

    resolved_split = split or (manifest.get("monitor_split", "test") if manifest else "test")
    requested_name = requested_view_name or (manifest.get("monitor_view_name", "") if manifest else "")
    if requested_name:
        normalized_name = normalize_image_name(requested_name)
        for idx, camera in enumerate(camera_list):
            if normalize_image_name(camera.image_name) == normalized_name:
                return resolved_split, idx, camera
        raise ValueError(f"Monitor view `{requested_name}` not found in `{resolved_split}` split.")

    manifest_index = int(manifest.get("monitor_index", requested_index)) if manifest else requested_index
    resolved_index = max(0, min(int(manifest_index), len(camera_list) - 1))
    return resolved_split, resolved_index, camera_list[resolved_index]


def monitor_dir(model_path: str, split: str, resolved_index: int, image_name: str) -> Path:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", normalize_image_name(image_name)).strip("_") or "view"
    return Path(model_path) / "monitor" / "fixed_views" / f"{split}_{resolved_index:04d}_{sanitized}"


def ensure_monitor_manifest(output_dir: Path, split: str, resolved_index: int, image_name: str, render_interval: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "renders").mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "resolved_index": int(resolved_index),
        "view_name": normalize_image_name(image_name),
        "render_interval": int(render_interval),
        "gt_filename": "gt.png",
        "renders_dir": "renders",
        "video_layout": "vertical",
        "video_fps": 8,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_tensor_image(tensor: torch.Tensor, output_path: Path):
    image = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(output_path)


def append_eval_history(model_path: str, record: Dict[str, Any]):
    monitor_root = Path(model_path) / "monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = monitor_root / "eval_history.jsonl"
    csv_path = monitor_root / "eval_history.csv"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    fieldnames = ["iteration", "split", "scope", "num_views", "l1", "psnr", "ssim", "num_gaussians"]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: record.get(name) for name in fieldnames})


def write_metrics_summary(model_path: str, payload: Dict[str, Any]):
    metrics_path = Path(model_path) / "metrics.json"
    normalized = dict(payload)
    if "l1" in normalized and "test_l1" not in normalized:
        normalized["test_l1"] = normalized["l1"]
    normalized.setdefault("status", "completed")
    metrics_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
