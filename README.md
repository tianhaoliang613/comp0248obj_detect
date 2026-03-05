# COMP0248 CW1 — Multi-Task Hand Gesture Understanding

Multi-task deep learning model for simultaneous hand segmentation, gesture classification, and bounding box detection from RGB(-D) images.

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is provided by the course instructor and is **not included** in this repository.
Expected structure under each data root:

```
<data_root>/
  <student_id>/
    rgb/        *.png  (RGB frames)
    depth_raw/  *.png  (16-bit depth frames, optional)
    masks/      *.png  (binary segmentation masks)
    labels.json        (gesture label per clip)
```

## Training

### RGB Baseline
```bash
python src/train.py \
  --data_roots <path_to_dataset1> <path_to_dataset2> \
  --backbone resnet18 --no_depth --amp \
  --epochs 60 --batch_size 8 --lr 5e-4 \
  --warmup_epochs 3 --label_smoothing 0.1 --patience 20 \
  --save_dir results/rgb_baseline/weights \
  --log_dir results/rgb_baseline
```

### Final Model (Mask-Guided + Staged Freeze)
Requires a pre-trained baseline checkpoint from the step above.

```bash
python src/train.py \
  --data_roots <path_to_dataset1> <path_to_dataset2> \
  --backbone resnet18 --no_depth --amp \
  --epochs 60 --batch_size 8 --lr 5e-4 \
  --warmup_epochs 3 --label_smoothing 0.1 --patience 20 \
  --maskguided --freeze_backbone_epochs 20 --backbone_lr_mult 0.02 \
  --init_checkpoint results/rgb_baseline/weights/best_resnet18_rgb.pth \
  --save_dir results/rgb_maskguided_freeze20/weights \
  --log_dir results/rgb_maskguided_freeze20
```

## Evaluation

```bash
python src/evaluate.py \
  --checkpoint results/rgb_maskguided_freeze20/weights/best_resnet18_rgb.pth \
  --data_roots <path_to_dataset1> <path_to_dataset2> \
  --split val --batch_size 8 \
  --save_dir results/rgb_maskguided_freeze20
```

## Source Files

| File | Description |
|------|-------------|
| `src/train.py` | Training loop with AMP, cosine LR schedule, early stopping |
| `src/evaluate.py` | Evaluation script (IoU, Dice, F1, detection accuracy) |
| `src/dataloader.py` | Dataset loading, augmentation, train/val split |
| `src/model.py` | ResNet encoder + U-Net decoder + classification head |
| `src/utils.py` | Metric utilities (IoU, Dice, bbox, confusion matrix) |
