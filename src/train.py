import os
import sys
import time
import json
import argparse
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import create_dataloaders, NUM_CLASSES, DEFAULT_DATA_ROOTS, PROJECT_ROOT, normalize_data_roots
from model import HandGestureModel, CombinedLoss


def parse_args():
    p = argparse.ArgumentParser("Simple training script (submission version)")
    p.add_argument("--data_roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    p.add_argument("--img_size", type=int, nargs=2, default=[480, 480])
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--use_depth", action="store_true", default=True)
    p.add_argument("--no_depth", dest="use_depth", action="store_false")
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--freeze_backbone_epochs", type=int, default=0)
    p.add_argument("--backbone_lr_mult", type=float, default=1.0)
    p.add_argument("--maskguided", action="store_true", default=False)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "weights"))
    p.add_argument("--log_dir", type=str, default=os.path.join(PROJECT_ROOT, "results"))
    return p.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seg_iou(seg_logits, mask):
    pred = (torch.sigmoid(seg_logits) > 0.5).float()
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - inter
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


def cls_acc(cls_logits, label):
    return (cls_logits.argmax(dim=1) == label).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False, train_mode=True):
    model.train() if train_mode else model.eval()
    total = {"loss": 0.0, "seg_loss": 0.0, "cls_loss": 0.0, "seg_iou": 0.0, "cls_acc": 0.0}
    n = 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, masks, labels, _ in loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            if train_mode and use_amp and scaler is not None:
                with autocast(device_type="cuda"):
                    seg_logits, cls_logits = model(images)
                    loss, s_loss, c_loss = criterion(seg_logits, cls_logits, masks, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                seg_logits, cls_logits = model(images)
                loss, s_loss, c_loss = criterion(seg_logits, cls_logits, masks, labels)
                if train_mode:
                    loss.backward()
                    optimizer.step()
            total["loss"] += loss.item()
            total["seg_loss"] += s_loss.item()
            total["cls_loss"] += c_loss.item()
            total["seg_iou"] += seg_iou(seg_logits.detach(), masks)
            total["cls_acc"] += cls_acc(cls_logits.detach(), labels)
            n += 1
    for k in total:
        total[k] /= max(n, 1)
    return total


def load_init_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    src = ckpt.get("model_state_dict", ckpt)
    dst = model.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(matched)
    model.load_state_dict(dst)
    print(f"[init_checkpoint] loaded {len(matched)} layers")


def train():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    args.data_roots = normalize_data_roots(args.data_roots)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        data_roots=args.data_roots, batch_size=args.batch_size, use_depth=args.use_depth,
        img_size=tuple(args.img_size), val_ratio=args.val_ratio, num_workers=args.num_workers, seed=args.seed
    )

    model = HandGestureModel(
        num_classes=NUM_CLASSES, in_channels=(4 if args.use_depth else 3),
        pretrained=args.pretrained, backbone=args.backbone, use_maskguided=args.maskguided
    ).to(device)
    if args.init_checkpoint:
        load_init_checkpoint(model, args.init_checkpoint, device)

    criterion = CombinedLoss(label_smoothing=args.label_smoothing).to(device)
    bb_prefix = ("encoder_conv1", "encoder_pool", "encoder_layer1", "encoder_layer2", "encoder_layer3", "encoder_layer4")
    bb_params, head_params = [], []
    for name, p in model.named_parameters():
        (bb_params if name.startswith(bb_prefix) else head_params).append(p)
    if args.freeze_backbone_epochs > 0:
        for p in bb_params:
            p.requires_grad = False
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs")

    optimizer = optim.AdamW(
        [{"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
         {"params": bb_params, "lr": args.lr * args.backbone_lr_mult, "weight_decay": args.weight_decay}],
        lr=args.lr
    )
    cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6)
    if args.warmup_epochs > 0:
        warm = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[args.warmup_epochs])
    else:
        scheduler = cos

    scaler = GradScaler("cuda") if args.amp and torch.cuda.is_available() else None
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="COMP0248-hand-gesture", name=args.wandb_run_name, config=vars(args))

    history = {"train": [], "val": []}
    best_score, best_epoch, best_val = float("-inf"), 0, None
    patience_cnt = 0
    mode_name = f"{args.backbone}_{'rgbd' if args.use_depth else 'rgb'}"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            for p in bb_params:
                p.requires_grad = True
            print("Backbone unfrozen")

        tr = run_epoch(model, train_loader, criterion, optimizer, device, scaler, args.amp, train_mode=True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, scaler, args.amp, train_mode=False)
        scheduler.step()
        score = 0.5 * va["seg_iou"] + 0.5 * va["cls_acc"]
        history["train"].append(tr); history["val"].append(va)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "train/loss": tr["loss"], "train/seg_loss": tr["seg_loss"], "train/cls_loss": tr["cls_loss"],
                "train/seg_iou": tr["seg_iou"], "train/cls_acc": tr["cls_acc"],
                "val/loss": va["loss"], "val/seg_loss": va["seg_loss"], "val/cls_loss": va["cls_loss"],
                "val/seg_iou": va["seg_iou"], "val/cls_acc": va["cls_acc"], "val/select_score": score,
            })

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({time.time()-t0:.1f}s) | "
            f"Sel {score:.4f} | Val IoU {va['seg_iou']:.4f} | Val Acc {va['cls_acc']:.4f} | Val Loss {va['loss']:.4f}"
        )

        if score > best_score:
            best_score, best_epoch, best_val = score, epoch, dict(va)
            patience_cnt = 0
            save_path = os.path.join(args.save_dir, f"best_{mode_name.lower()}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val,
                "args": vars(args),
            }, save_path)
            print(f"  >> Best saved ({best_score:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break

    log_path = os.path.join(args.log_dir, f"training_log_{mode_name.lower()}.json")
    json.dump(history, open(log_path, "w"), indent=2)
    print("Training log:", log_path)
    if best_epoch > 0:
        print(f"Best epoch={best_epoch}, best joint={best_score:.4f}, val={best_val}")

    if wandb_run is not None:
        wandb_run.summary["best_joint"] = best_score
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


if __name__ == "__main__":
    train()
