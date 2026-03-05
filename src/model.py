"""Model definition: ResNet encoder + U-Net decoder + mask-guided classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np


# ============================================================
# Decoder upsampling block
# ============================================================
class DecoderBlock(nn.Module):
    """U-Net decoder block: upsample + skip concat + two Conv-BN-ReLU."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Concatenated channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ============================================================
# Multi-task model
# ============================================================
class HandGestureModel(nn.Module):
    """Multi-task hand model with mask-guided classification."""
    
    def __init__(self, num_classes=10, in_channels=4, pretrained=True, backbone="resnet18", use_maskguided=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_maskguided = use_maskguided
        self.backbone_name = backbone.lower()
        if self.backbone_name not in {"resnet18", "resnet50"}:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet50'.")

        if self.backbone_name == "resnet18":
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18(weights=None)
            enc_c2, enc_c3, enc_c4, enc_c5 = 64, 128, 256, 512
        else:
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                resnet = models.resnet50(weights=None)
            enc_c2, enc_c3, enc_c4, enc_c5 = 256, 512, 1024, 2048
        
        # Adjust first conv for RGBD input.
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                if in_channels > 3:
                    new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv
        
        # Split ResNet layers to extract intermediate features for skip connections
        self.encoder_conv1 = nn.Sequential(
            resnet.conv1,   # stride=2, output (B, 64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
        )
        self.encoder_pool = resnet.maxpool   # stride=2, output (B, 64, H/4, W/4)
        self.encoder_layer1 = resnet.layer1  # output (B, c2, H/4,  W/4)
        self.encoder_layer2 = resnet.layer2  # output (B, c3, H/8,  W/8)
        self.encoder_layer3 = resnet.layer3  # output (B, c4, H/16, W/16)
        self.encoder_layer4 = resnet.layer4  # output (B, c5, H/32, W/32)
        
        self.decoder4 = DecoderBlock(enc_c5, enc_c4, 256)  # F5 + F4 → 256
        self.decoder3 = DecoderBlock(256, enc_c3, 128)     # + F3 → 128
        self.decoder2 = DecoderBlock(128, enc_c2, 64)      # + F2 → 64
        self.decoder1 = DecoderBlock(64, 64, 64)           # + F1 → 64
        
        # Final upsampling + 1x1 conv to output mask
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # output 1 channel (foreground probability)
        
        self.cls_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(enc_c5, num_classes)
        )
    
    def forward(self, x):
        input_size = x.shape[2:]

        f1 = self.encoder_conv1(x)          # (B, 64, H/2, W/2)
        f1_pooled = self.encoder_pool(f1)   # (B, 64, H/4, W/4)
        f2 = self.encoder_layer1(f1_pooled) # (B, c2, H/4, W/4)
        f3 = self.encoder_layer2(f2)        # (B, c3, H/8, W/8)
        f4 = self.encoder_layer3(f3)        # (B, c4, H/16, W/16)
        f5 = self.encoder_layer4(f4)        # (B, c5, H/32, W/32)

        d4 = self.decoder4(f5, f4)  # (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, f3)  # (B, 128, H/8,  W/8)
        d2 = self.decoder2(d3, f2)  # (B, 64,  H/4,  W/4)
        d1 = self.decoder1(d2, f1)  # (B, 64,  H/2,  W/2)

        seg_logits = self.final_conv(d1)
        seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)

        if self.use_maskguided:
            pred_mask = torch.sigmoid(seg_logits.detach())
            mask_small = F.interpolate(pred_mask, size=f5.shape[2:], mode='bilinear', align_corners=False)
            weighted_sum = (f5 * mask_small).sum(dim=[2, 3])
            mask_sum = mask_small.sum(dim=[2, 3]).clamp(min=1.0)
            cls_features = weighted_sum / mask_sum
        else:
            cls_features = F.adaptive_avg_pool2d(f5, (1, 1)).view(f5.size(0), -1)

        cls_logits = self.cls_head(cls_features)

        return seg_logits, cls_logits


# ============================================================
# Bounding box extraction (non-learnable, post-processing)
# ============================================================
def mask_to_bbox(mask_np):
    """
    Extract bounding box from binary mask.
    
    This is a pure post-processing function, not involved in training.
    Used during evaluation and inference.

    Args:
        mask_np: numpy array, shape (H, W), values 0 or 1 (or 0/255)

    Returns:
        bbox: (x1, y1, x2, y2) or None (if mask is all zeros)
    """
    # Ensure mask is uint8
    if mask_np.dtype != np.uint8:
        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = mask_np
    
    # Find bounding rectangle of non-zero pixels
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Merge bounding rectangles of all contours
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)
    
    return (x, y, x + w, y + h)  # (x1, y1, x2, y2)


# ============================================================
# Loss functions
# ============================================================
class CombinedLoss(nn.Module):
    """Multi-task loss: lambda_seg*(BCE+Dice) + lambda_cls*CE."""
    
    def __init__(self, lambda_seg=1.0, lambda_cls=1.0, label_smoothing=0.0):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def dice_loss(self, pred_logits, target):
        smooth = 1.0
        pred = torch.sigmoid(pred_logits)
        
        pred_flat = pred.view(pred.size(0), -1)       # (B, H*W)
        target_flat = target.view(target.size(0), -1)  # (B, H*W)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def forward(self, seg_logits, cls_logits, mask_target, cls_target):
        bce_loss = self.bce(seg_logits, mask_target)
        dice_loss = self.dice_loss(seg_logits, mask_target)
        seg_loss = bce_loss + dice_loss
        cls_loss = self.ce(cls_logits, cls_target)
        total_loss = self.lambda_seg * seg_loss + self.lambda_cls * cls_loss
        return total_loss, seg_loss, cls_loss

