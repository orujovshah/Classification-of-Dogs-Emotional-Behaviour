import os
import random
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torch.utils.data import Dataset

import timm

from helpers import (
    set_seed,
    make_dataloader,
    train_validate,
    save_curve,
    run_model,
    generate_eigencam_overlays
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR       = "../dataset"
IMG_SIZE       = 224    # reduced resolution to save memory
BATCH_SIZE     = 16
NUM_WORKERS    = 4
TEST_FRAC      = 0.15
TRAIN_FRAC     = 0.70
OUTPUT_DIR     = "./outputs_experiment_models"
SEED           = 42

STAGE1_EPOCHS  = 10
STAGE2_EPOCHS  = 10
STAGE3_EPOCHS  = 30
EARLY_STOPPING = None   # or set an int for patience

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

# ─── TRANSFORMS & DATASETS ────────────────────────────────────────────────────
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
        for idx, (_path, lab) in enumerate(self.sub):
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

# ─── DATALOADERS ──────────────────────────────────────────────────────────────
train_loader = make_dataloader(
    SMOTEDataset(all_samples, train_idxs, train_tf, p_synth=0.7, alpha=0.2),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, seed=SEED
)
val_loader = make_dataloader(
    FourDataset(all_samples, val_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, seed=SEED
)
test_loader = make_dataloader(
    FourDataset(all_samples, test_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, seed=SEED
)

# ─── UTILITY ─────────────────────────────────────────────────────────────────
def evaluate(name, model, history):
    mdir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(mdir, exist_ok=True)
    save_curve(history['train_acc'], history['val_acc'], name, "Acc", mdir)
    save_curve(history['train_loss'], history['val_loss'], name, "Loss", mdir)
    run_model(name, model, history, test_loader, class_names, mdir, device)
    generate_eigencam_overlays(model, test_loader.dataset, class_names, name, device, mdir)
    torch.cuda.empty_cache()

# ─── A) EfficientNetV2-M ──────────────────────────────────────────────────────
print("\n=== Training EfficientNetV2-M ===")
set_seed(SEED)
eff = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True, drop_path_rate=0.3).to(device)
object.__setattr__(eff, 'features', eff.blocks)
for p in eff.parameters(): p.requires_grad = False

# rebuild head
in_feats = eff.classifier.in_features
eff.classifier = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(in_feats, len(class_names))
).to(device)

# class weights + label smoothing
train_labels = torch.tensor([all_labels[i] for i in train_idxs], device=device)
counts = torch.bincount(train_labels, minlength=len(class_names)).float()
weights = counts.sum() / (len(class_names) * counts)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# Stage 1
for p in eff.classifier.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, eff.parameters()), lr=1e-4, weight_decay=0.1)
hist1 = train_validate(
    eff, opt, criterion, STAGE1_EPOCHS, "Eff-M-Stage1",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=OUTPUT_DIR, early_stopping_patience=EARLY_STOPPING
)

# Stage 2
for blk in eff.blocks[-3:]:
    for p in blk.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, eff.parameters()), lr=5e-5, weight_decay=0.1)
sched = CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)

tr2, vl2, tl2, vl2l = [], [], [], []
for ep in range(STAGE2_EPOCHS):
    h = train_validate(eff, opt, criterion, 1, f"Eff-M-Stage2-Ep{ep+1}",
                       train_loader, val_loader, device, class_names)
    tr2.append(h['train_acc'][0]); vl2.append(h['val_acc'][0])
    tl2.append(h['train_loss'][0]); vl2l.append(h['val_loss'][0])
    sched.step()

# Stage 3
for p in eff.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, eff.parameters()), lr=1e-5, weight_decay=0.1)
sched = CosineAnnealingLR(opt, T_max=STAGE3_EPOCHS)

tr3, vl3, tl3, vl3l = [], [], [], []
for ep in range(STAGE3_EPOCHS):
    h = train_validate(eff, opt, criterion, 1, f"Eff-M-Stage3-Ep{ep+1}",
                       train_loader, val_loader, device, class_names)
    tr3.append(h['train_acc'][0]); vl3.append(h['val_acc'][0])
    tl3.append(h['train_loss'][0]); vl3l.append(h['val_loss'][0])
    sched.step()

history_eff = {
    'train_acc':  hist1['train_acc']  + tr2  + tr3,
    'val_acc':    hist1['val_acc']    + vl2  + vl3,
    'train_loss': hist1['train_loss'] + tl2  + tl3,
    'val_loss':   hist1['val_loss']   + vl2l + vl3l,
}
evaluate("EfficientNetV2-M", eff, history_eff)


# ─── B) ConvNeXt-Base ─────────────────────────────────────────────────────────
print("\n=== Training ConvNeXt-Base ===")
set_seed(SEED)
cn = timm.create_model(
    'convnext_base',
    pretrained=True,
    drop_path_rate=0.3,
    num_classes=len(class_names)     # ensure classifier matches our classes
).to(device)
object.__setattr__(cn, 'features', cn.stages)
for p in cn.parameters(): p.requires_grad = False

# unfreeze head
for p in cn.head.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, cn.parameters()), lr=1e-4, weight_decay=0.1)
hist1 = train_validate(
    cn, opt, criterion, STAGE1_EPOCHS, "CN-Base-Stage1",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=OUTPUT_DIR, early_stopping_patience=EARLY_STOPPING
)

# Stage 2
for stage in cn.stages[-3:]:
    for p in stage.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, cn.parameters()), lr=5e-5, weight_decay=0.1)
sched = CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)

tr2, vl2, tl2, vl2l = [], [], [], []
for ep in range(STAGE2_EPOCHS):
    h = train_validate(cn, opt, criterion, 1, f"CN-Base-Stage2-Ep{ep+1}",
                       train_loader, val_loader, device, class_names)
    tr2.append(h['train_acc'][0]); vl2.append(h['val_acc'][0])
    tl2.append(h['train_loss'][0]); vl2l.append(h['val_loss'][0])
    sched.step()

# Stage 3
for p in cn.parameters(): p.requires_grad = True
opt = AdamW(filter(lambda p: p.requires_grad, cn.parameters()), lr=1e-5, weight_decay=0.1)
sched = CosineAnnealingLR(opt, T_max=STAGE3_EPOCHS)

tr3, vl3, tl3, vl3l = [], [], [], []
for ep in range(STAGE3_EPOCHS):
    h = train_validate(cn, opt, criterion, 1, f"CN-Base-Stage3-Ep{ep+1}",
                       train_loader, val_loader, device, class_names)
    tr3.append(h['train_acc'][0]); vl3.append(h['val_acc'][0])
    tl3.append(h['train_loss'][0]); vl3l.append(h['val_loss'][0])
    sched.step()

history_cn = {
    'train_acc':  hist1['train_acc']  + tr2  + tr3,
    'val_acc':    hist1['val_acc']    + vl2  + vl3,
    'train_loss': hist1['train_loss'] + tl2  + tl3,
    'val_loss':   hist1['val_loss']   + vl2l + vl3l,
}
evaluate("ConvNeXt-Base", cn, history_cn)

print("\n✅ Conv models done. Outputs in", OUTPUT_DIR)
