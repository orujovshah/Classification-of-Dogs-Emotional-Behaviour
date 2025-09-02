import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def dino_vit_reshape_transform(tokens, image_h=224, image_w=224, patch=14):
    # tokens: (B, N+1, C). Drop CLS and reshape to (B, C, H, W)
    B, Np1, C = tokens.shape
    H = image_h // patch
    W = image_w // patch
    patch_tokens = tokens[:, 1:, :]                    # (B, N, C)
    return patch_tokens.transpose(1, 2).reshape(B, C, H, W)

def pick_vit_target_layers(backbone):
    # Late blocks are most semantic; you can average several if you like.
    return [backbone.blocks[-1].norm1]

def is_vit_like(m):
    return hasattr(m, "blocks") and hasattr(m, "patch_embed")

def generate_eigencam_overlays(
    model, dataset, class_names,
    model_name, device, output_dir,
    mean=None, std=None, img_size=224, patch=14
):
    if mean is None: mean = [0.485,0.456,0.406]
    if std  is None: std  = [0.229,0.224,0.225]

    # If the model is nn.Sequential(backbone, head), grab the backbone for target layers.
    backbone = model[0] if isinstance(model, nn.Sequential) else model

    use_vit = is_vit_like(backbone)
    if use_vit:
        target_layers = pick_vit_target_layers(backbone)
        reshape = lambda t: dino_vit_reshape_transform(t, img_size, img_size, patch)
    else:
        # fallback: last conv for CNNs
        target_layers = [next(m for m in reversed(list(model.modules()))
                              if isinstance(m, nn.Conv2d))]
        reshape = None

    model.to(device).eval()

    # pick first correct + incorrect per class
    correct = {i:None for i in range(len(class_names))}
    incorrect = {i:None for i in range(len(class_names))}
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            pred = model(inp).argmax(1).item()
            if pred == label and correct[label] is None:
                correct[label] = (idx, pred)
            if pred != label and incorrect[label] is None:
                incorrect[label] = (idx, pred)
            if all(correct.values()) and all(incorrect.values()):
                break

    os.makedirs(output_dir, exist_ok=True)

    # Build a single CAM object and reuse it (faster, less VRAM churn)
    cam = EigenCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape
    )

    def _build(bucket, kind):
        imgs, titles = [], []
        for cls_idx, entry in bucket.items():
            if entry is None: continue
            idx, pred = entry
            img, label = dataset[idx]
            inp = img.unsqueeze(0).to(device)

            # EigenCAM doesn’t need targets; it’s gradient-free
            gcam = cam(input_tensor=inp)[0]           # (H, W), upsampled to input

            arr = img.cpu().numpy().transpose(1,2,0)
            arr = (arr * np.array(std)) + np.array(mean)
            arr = np.clip(arr, 0, 1)
            overlay = show_cam_on_image(arr, gcam, use_rgb=True)
            imgs.append(overlay)
            titles.append(f"{class_names[label]}\n(pred={class_names[pred]})")

        if not imgs: return
        fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs),4))
        if len(imgs) == 1: axes = [axes]
        for ax, im, tt in zip(axes, imgs, titles):
            ax.imshow(im); ax.set_title(tt, fontsize=9); ax.axis('off')
        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(output_dir, f"{model_name}_{kind}_overlays.svg"),
                    format='svg', bbox_inches='tight')
        plt.close()

    _build(correct,   "correct")
    _build(incorrect, "incorrect")
