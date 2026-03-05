import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOTS = [
    os.path.join(PROJECT_ROOT, "OneDrive_dataset", "RGB_depth_annotations"),
    os.path.join(PROJECT_ROOT, "OneDrive_dataset", "late submission"),
]
GESTURES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]
GESTURE_TO_LABEL = {g: i for i, g in enumerate(GESTURES)}
NUM_CLASSES = len(GESTURES)


def normalize_data_roots(data_roots):
    out = []
    for p in data_roots:
        if not p:
            continue
        out.append(p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p)))
    return list(dict.fromkeys(out))


def discover_student_data(data_roots):
    dirs = []
    for root in data_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__MACOSX"]
            if "G01_call" in dirnames:
                if sum(1 for g in GESTURES if g in dirnames) >= 8:
                    dirs.append(dirpath)
    dirs = sorted(set(dirs))
    print(f"[INFO] found {len(dirs)} student dirs")
    return dirs


def build_sample_list(student_dirs):
    samples = []
    for sdir in student_dirs:
        for gesture in GESTURES:
            gdir = os.path.join(sdir, gesture)
            if not os.path.isdir(gdir):
                continue
            for clip in sorted(os.listdir(gdir)):
                cdir = os.path.join(gdir, clip)
                if not os.path.isdir(cdir):
                    continue
                ann_dir = os.path.join(cdir, "annotation")
                rgb_dir = os.path.join(cdir, "rgb")
                depth_dir = os.path.join(cdir, "depth_raw")
                meta_path = os.path.join(cdir, "depth_metadata.json")
                if not os.path.isdir(ann_dir):
                    continue
                depth_scale = 0.001
                if os.path.isfile(meta_path):
                    try:
                        depth_scale = json.load(open(meta_path, "r")).get("depth_scale", 0.001)
                    except Exception:
                        pass
                for mf in sorted(os.listdir(ann_dir)):
                    if (not mf.endswith(".png")) or mf.startswith("._"):
                        continue
                    rgb_path = os.path.join(rgb_dir, mf)
                    depth_path = os.path.join(depth_dir, mf.replace(".png", ".npy"))
                    if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)):
                        continue
                    samples.append({
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "mask_path": os.path.join(ann_dir, mf),
                        "gesture": gesture,
                        "label": GESTURE_TO_LABEL[gesture],
                        "student_id": sdir,
                        "depth_scale": depth_scale,
                    })
    print(f"[INFO] built {len(samples)} samples")
    return samples


def split_by_student(samples, val_ratio=0.15, seed=42):
    ids = sorted(set(x["student_id"] for x in samples))
    random.seed(seed)
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = [x for x in samples if x["student_id"] not in val_ids]
    val = [x for x in samples if x["student_id"] in val_ids]
    print(f"[INFO] split: train {len(train)} / val {len(val)}")
    return train, val


class HandGestureDataset(Dataset):
    def __init__(self, samples, use_depth=True, img_size=(480, 480), augment=False):
        self.samples = samples
        self.use_depth = use_depth
        self.img_size = img_size
        self.augment = augment
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        bgr = cv2.imread(s["rgb_path"])
        if bgr is None:
            raise FileNotFoundError(s["rgb_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(s["depth_path"]).astype(np.float32) * s["depth_scale"]
        mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(s["mask_path"])
        mask = (mask > 127).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        h, w = self.img_size
        rgb_t = F.resize(rgb_t, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        depth_t = F.resize(depth_t, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_t = F.resize(mask_t, [h, w], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            rgb_t, depth_t, mask_t = self._augment(rgb_t, depth_t, mask_t)

        rgb_t = F.normalize(rgb_t.float() / 255.0, mean=self.rgb_mean, std=self.rgb_std)
        depth_t = torch.clamp(depth_t, 0.0, 3.0) / 3.0
        image = torch.cat([rgb_t, depth_t], dim=0) if self.use_depth else rgb_t
        return image, mask_t.float(), s["label"], {"gesture": s["gesture"], "student_id": s["student_id"]}

    def _augment(self, rgb_t, depth_t, mask_t):
        if torch.rand(1).item() < 0.5:
            angle = random.uniform(-15, 15)
            rgb_t = F.rotate(rgb_t, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            depth_t = F.rotate(depth_t, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask_t = F.rotate(mask_t, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        if torch.rand(1).item() < 0.3:
            tx = int(rgb_t.shape[2] * random.uniform(-0.1, 0.1))
            ty = int(rgb_t.shape[1] * random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)
            rgb_t = F.affine(rgb_t, 0, [tx, ty], scale, [0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
            depth_t = F.affine(depth_t, 0, [tx, ty], scale, [0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
            mask_t = F.affine(mask_t, 0, [tx, ty], scale, [0.0], interpolation=InterpolationMode.NEAREST, fill=0)
        if torch.rand(1).item() < 0.3:
            h, w = rgb_t.shape[1], rgb_t.shape[2]
            sc = random.uniform(0.85, 1.0)
            ch, cw = max(1, int(h * sc)), max(1, int(w * sc))
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            rgb_t = F.resized_crop(rgb_t, top, left, ch, cw, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
            depth_t = F.resized_crop(depth_t, top, left, ch, cw, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask_t = F.resized_crop(mask_t, top, left, ch, cw, [h, w], interpolation=InterpolationMode.NEAREST)
        rgb_t = self.color_jitter(rgb_t)
        if torch.rand(1).item() < 0.2:
            rgb_t = F.gaussian_blur(rgb_t, kernel_size=5, sigma=(0.1, 2.0))
        if torch.rand(1).item() < 0.3:
            depth_t = torch.clamp(depth_t + torch.randn_like(depth_t) * random.uniform(0.003, 0.015), min=0.0)
        return rgb_t, depth_t, mask_t


def _collate_fn(batch):
    images, masks, labels, metas = zip(*batch)
    return torch.stack(images), torch.stack(masks), torch.tensor(labels, dtype=torch.long), list(metas)


def create_dataloaders(data_roots, batch_size=8, use_depth=True, img_size=(480, 480),
                       val_ratio=0.15, num_workers=4, seed=42):
    roots = normalize_data_roots(data_roots)
    students = discover_student_data(roots)
    samples = build_sample_list(students)
    train_samples, val_samples = split_by_student(samples, val_ratio=val_ratio, seed=seed)
    train_set = HandGestureDataset(train_samples, use_depth=use_depth, img_size=img_size, augment=True)
    val_set = HandGestureDataset(val_samples, use_depth=use_depth, img_size=img_size, augment=False)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=_collate_fn
    )
    return train_loader, val_loader
