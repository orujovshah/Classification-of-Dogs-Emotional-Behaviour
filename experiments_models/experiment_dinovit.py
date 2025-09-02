import os
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms

import timm

from helpers import (
    set_seed,
    make_dataloader,
    train_validate,
    save_curve,
    run_model,
)
from eigencam_dinovit_helper import generate_eigencam_overlays

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR       = "../dataset"
IMG_SIZE       = 224
BATCH_SIZE     = 16
NUM_WORKERS    = 4
TEST_FRAC      = 0.15
TRAIN_FRAC     = 0.70
OUTPUT_DIR     = "./outputs_experiment_dinovit"
SEED           = 42

STAGE1_EPOCHS  = 10   # head only
STAGE2_EPOCHS  = 10   # + last 3 ViT blocks
STAGE3_EPOCHS  = 30   # + full backbone
EARLY_STOPPING = None # disable early stopping

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── DATA SPLIT ───────────────────────────────────────────────────────────────
full_ds     = __import__('torchvision.datasets').datasets.ImageFolder(BASE_DIR, transform=None)
class_names = full_ds.classes
all_samples = full_ds.samples
all_labels  = [lab for (_p, lab) in all_samples]

idxs_per_cls = defaultdict(list)
for i, (_p, lab) in enumerate(all_samples):
    idxs_per_cls[lab].append(i)

train_idxs, val_idxs, test_idxs = [], [], []
for lab, idxs in idxs_per_cls.items():
    random.seed(SEED + lab)
    random.shuffle(idxs)
    n  = len(idxs)
    nt = max(1, int(n * TEST_FRAC))
    test_idxs.extend(idxs[:nt])
    rem = idxs[nt:]
    nv  = int(len(rem) * TRAIN_FRAC)
    train_idxs.extend(rem[:nv])
    val_idxs.extend(rem[nv:])

print(f"Splits → train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")

# ─── TRANSFORMS & DATALOADERS ─────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class SMOTEDataset(Dataset):
    def __init__(self, samples, indices, tf, p_synth=0.5, alpha=0.4):
        self.sub     = [samples[i] for i in indices]
        self.tf      = tf
        self.p_synth = p_synth
        self.alpha   = alpha
        self.local_map = defaultdict(list)
        for idx, (_p, lab) in enumerate(self.sub):
            self.local_map[lab].append(idx)

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, i):
        path1, label = self.sub[i]
        img1 = self.tf(default_loader(path1).convert("RGB"))
        if random.random() < self.p_synth:
            j = random.choice(self.local_map[label])
            path2, _ = self.sub[j]
            img2 = self.tf(default_loader(path2).convert("RGB"))
            lam = np.random.beta(self.alpha, self.alpha)
            img = lam * img1 + (1 - lam) * img2
            return img, label
        return img1, label

class FourDataset(Dataset):
    def __init__(self, samples, indices, tf):
        self.sub = [samples[i] for i in indices]
        self.tf  = tf

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, i):
        p, lab = self.sub[i]
        img    = default_loader(p).convert("RGB")
        return self.tf(img), lab

train_loader = make_dataloader(
    SMOTEDataset(all_samples, train_idxs, train_tf, p_synth=0.7, alpha=0.2),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, seed=SEED
)
val_loader   = make_dataloader(
    FourDataset(all_samples, val_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, seed=SEED
)
test_loader  = make_dataloader(
    FourDataset(all_samples, test_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, seed=SEED
)

# ─── CLASS‐WEIGHTED LOSS ───────────────────────────────────────────────────────
train_labels = torch.tensor([all_labels[i] for i in train_idxs], device=device)
counts       = torch.bincount(train_labels, minlength=len(class_names)).float()
weights      = counts.sum() / (len(class_names) * counts)
criterion    = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# ─── MODEL UTILITY ────────────────────────────────────────────────────────────
def evaluate(name, model, history):
    mdir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(mdir, exist_ok=True)
    save_curve(history['train_acc'], history['val_acc'], name, "Acc", mdir)
    save_curve(history['train_loss'], history['val_loss'], name, "Loss", mdir)
    run_model(name, model, history, test_loader, class_names, mdir, device)
    generate_eigencam_overlays(model, test_loader.dataset, class_names, name, device, mdir)
    torch.cuda.empty_cache()

# ─── DINO‐ViT EXPERIMENT ──────────────────────────────────────────────────────
print("\n=== Training DINO-ViT-S ===")
set_seed(SEED)
# load DINOv2-small backbone
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# build head
head = nn.Sequential(
    nn.LayerNorm(backbone.embed_dim, eps=1e-6),
    nn.Linear(backbone.embed_dim, len(class_names))
).to(device)
# combine into a single nn.Module
model = nn.Sequential(backbone, head).to(device)
# tell CAM helper where to hook
import torch.nn as _nn
object.__setattr__(model, 'features', _nn.ModuleList([backbone.patch_embed.proj]))

# Stage 1: head only
for p in backbone.parameters(): p.requires_grad = False
for p in head.parameters():     p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.1)
hist1 = train_validate(
    model, opt, criterion, STAGE1_EPOCHS, "DINO-S-Stage1",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=OUTPUT_DIR, early_stopping_patience=EARLY_STOPPING
)

# Stage 2: unfreeze last 3 transformer blocks
for blk in backbone.blocks[-3:]:
    for p in blk.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.1)
sched = CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
tr2, vl2, tl2, vl2l = [], [], [], []
for ep in range(STAGE2_EPOCHS):
    h = train_validate(
        model, opt, criterion, 1, f"DINO-S-Stage2-Ep{ep+1}",
        train_loader, val_loader, device, class_names
    )
    tr2 .append(h['train_acc'][0])
    vl2 .append(h['val_acc'][0])
    tl2 .append(h['train_loss'][0])
    vl2l.append(h['val_loss'][0])
    sched.step()

# Stage 3: unfreeze entire backbone
for p in backbone.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.1)
sched = CosineAnnealingLR(opt, T_max=STAGE3_EPOCHS)
tr3, vl3, tl3, vl3l = [], [], [], []
for ep in range(STAGE3_EPOCHS):
    h = train_validate(
        model, opt, criterion, 1, f"DINO-S-Stage3-Ep{ep+1}",
        train_loader, val_loader, device, class_names
    )
    tr3 .append(h['train_acc'][0])
    vl3 .append(h['val_acc'][0])
    tl3 .append(h['train_loss'][0])
    vl3l.append(h['val_loss'][0])
    sched.step()

# combine histories and evaluate
history = {
    'train_acc':  hist1['train_acc']  + tr2  + tr3,
    'val_acc':    hist1['val_acc']    + vl2  + vl3,
    'train_loss': hist1['train_loss'] + tl2  + tl3,
    'val_loss':   hist1['val_loss']   + vl2l + vl3l,
}
evaluate("DINO-ViT-S", model, history)

print("\n✅ DINO-ViT experiment done. Outputs in", OUTPUT_DIR)
