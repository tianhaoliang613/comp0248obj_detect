# Hand Gesture Recognition Project

This directory contains the core code for multi-task hand gesture understanding (segmentation, classification, and detection).

## Directory Structure

- **Training**: `src/train.py`
- **Evaluation**: `src/evaluate.py`
- **Data Loading**: `src/dataloader.py`
- **Model and Loss**: `src/model.py`
- **Metrics Utilities**: `src/utils.py`
- **Visualization**: `src/visualise.py`

`src_submit/` is a stripped-down submission version without visualization and extra utilities.

Note: `src/train.py` runs the standard baseline by default (mask-guided disabled); add `--maskguided` to enable mask-guided classification.

## Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Examples

### RGB baseline
```bash
python src/train.py \
  --data_roots <path_to_RGB_depth_annotations> <path_to_late_submission> \
  --backbone resnet18 --no_depth --amp --epochs 60 --batch_size 8 --lr 5e-4 \
  --warmup_epochs 3 --label_smoothing 0.1 --patience 20 \
  --save_dir results/experiments/rgb_baseline/weights \
  --log_dir results/experiments/rgb_baseline
```

### Final Model (mask-guided + freeze)
```bash
python src/train.py \
  --data_roots <path_to_RGB_depth_annotations> <path_to_late_submission> \
  --backbone resnet18 --no_depth --amp --epochs 60 --batch_size 8 --lr 5e-4 \
  --warmup_epochs 3 --label_smoothing 0.1 --patience 20 \
  --maskguided --freeze_backbone_epochs 20 --backbone_lr_mult 0.02 \
  --init_checkpoint results/experiments/resnet18_rgb_baseline/best_resnet18_rgb.pth \
  --save_dir results/experiments/rgb_maskguided_freeze20/weights \
  --log_dir results/experiments/rgb_maskguided_freeze20
```

## Evaluation Example
```bash
python src/evaluate.py \
  --checkpoint results/experiments/rgb_maskguided_freeze20/weights/best_resnet18_rgb.pth \
  --split val --batch_size 8 --num_workers 4 \
  --save_dir results/experiments/rgb_maskguided_freeze20
```

## Output Directories

- **`weights/`**: Trained model weight files (.pth)
- **`results/`**: Prediction results, logs, and generated charts (loss curves, confusion matrices, etc.)
- **`figures/`**: Figures used in the report
- **`report/`**: LaTeX report files
