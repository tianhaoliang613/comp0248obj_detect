import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import (
    create_dataloaders, HandGestureDataset, discover_student_data, build_sample_list,
    _collate_fn, GESTURES, NUM_CLASSES, DEFAULT_DATA_ROOTS, PROJECT_ROOT, normalize_data_roots
)
from model import HandGestureModel
from utils import compute_iou, compute_dice, compute_bbox_iou, mask_to_bbox, compute_confusion_matrix, compute_f1_macro


def parse_args():
    p = argparse.ArgumentParser("Simple evaluation script (submission version)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--test_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "results"))
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    seg_ious, dices, bbox_ious, det_ok = [], [], [], []
    preds, labels = [], []
    for images, masks, y, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        seg_logits, cls_logits = model(images)
        seg_pred = (torch.sigmoid(seg_logits).cpu().numpy() > 0.5).astype(np.float32)
        cls_pred = cls_logits.argmax(dim=1).cpu().numpy()
        masks_np = masks.numpy()
        y_np = y.numpy()
        for i in range(images.size(0)):
            pm, gm = seg_pred[i, 0], masks_np[i, 0]
            seg_ious.append(compute_iou(pm, gm))
            dices.append(compute_dice(pm, gm))
            pb, gb = mask_to_bbox(pm), mask_to_bbox(gm)
            if pb is None or gb is None:
                bbox_ious.append(0.0); det_ok.append(False)
            else:
                bi = compute_bbox_iou(pb, gb)
                bbox_ious.append(bi); det_ok.append(bi >= 0.5)
            preds.append(int(cls_pred[i]))
            labels.append(int(y_np[i]))

    cm = compute_confusion_matrix(preds, labels, NUM_CLASSES)
    macro_f1, per_class_f1 = compute_f1_macro(cm)
    return {
        "segmentation": {"mean_iou": float(np.mean(seg_ious)), "mean_dice": float(np.mean(dices))},
        "detection": {"accuracy_at_0.5_iou": float(np.mean(det_ok)), "mean_bbox_iou": float(np.mean(bbox_ious))},
        "classification": {
            "top1_accuracy": float(np.mean(np.array(preds) == np.array(labels))),
            "macro_f1": float(macro_f1),
            "per_class_f1": {GESTURES[i]: float(v) for i, v in enumerate(per_class_f1)},
            "confusion_matrix": cm.tolist(),
        },
        "num_samples": len(seg_ious),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    args.data_roots = normalize_data_roots(args.data_roots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    use_depth = saved_args.get("use_depth", True)
    backbone = saved_args.get("backbone", "resnet18")
    mg_arg = saved_args.get("maskguided", None)
    if mg_arg is None:
        use_maskguided = "maskguided" in args.checkpoint.lower()
    else:
        use_maskguided = bool(mg_arg)
    img_size = tuple(saved_args.get("img_size", [480, 480]))
    model = HandGestureModel(
        num_classes=NUM_CLASSES, in_channels=(4 if use_depth else 3),
        pretrained=False, backbone=backbone, use_maskguided=use_maskguided
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.split == "val":
        _, loader = create_dataloaders(
            data_roots=args.data_roots, batch_size=args.batch_size, use_depth=use_depth,
            img_size=img_size, val_ratio=saved_args.get("val_ratio", 0.15),
            num_workers=args.num_workers, seed=saved_args.get("seed", 42)
        )
    else:
        if args.test_root is None:
            raise ValueError("--test_root is required when --split test")
        students = discover_student_data([args.test_root])
        samples = build_sample_list(students)
        ds = HandGestureDataset(samples, use_depth=use_depth, img_size=img_size, augment=False)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate_fn
        )

    results = evaluate(model, loader, device)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"eval_{args.split}.json")
    json.dump(results, open(save_path, "w"), indent=2)

    print("Samples:", results["num_samples"])
    print("SegIoU:", f"{results['segmentation']['mean_iou']:.4f}",
          "Dice:", f"{results['segmentation']['mean_dice']:.4f}")
    print("Det@0.5:", f"{results['detection']['accuracy_at_0.5_iou']:.4f}",
          "BBoxIoU:", f"{results['detection']['mean_bbox_iou']:.4f}")
    print("Top1:", f"{results['classification']['top1_accuracy']:.4f}",
          "MacroF1:", f"{results['classification']['macro_f1']:.4f}")
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
