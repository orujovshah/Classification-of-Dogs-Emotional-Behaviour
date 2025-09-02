import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from helpers import (
    set_seed,
    make_dataloader,
    train_validate,
    save_curve,
    run_model,
    generate_eigencam_overlays
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR       = "./dataset"
IMG_SIZE       = 224
BATCH_SIZE     = 64
NUM_WORKERS    = 4
EPOCHS         = 50

OUTPUT_PARENT  = "./outputs_paperfaithful"
MODEL_SUBFOLD  = "3-class MobileNetV2"
OUTPUT_DIR     = os.path.join(OUTPUT_PARENT, MODEL_SUBFOLD)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── SPLIT FRACTIONS ──────────────────────────────────────────────────────────
TEST_FRAC   = 0.15   # 15% per original class for final test
TRAIN_FRAC  = 0.70   # of the remaining 85%, 70% train / 30% val

# ─── LOAD & STRATIFIED SPLIT ──────────────────────────────────────────────────
full_ds     = ImageFolder(BASE_DIR, transform=None)
all_samples = full_ds.samples
all_labels  = [lab for (_p, lab) in all_samples]

idxs_per_cls = defaultdict(list)
for i, lab in enumerate(all_labels):
    idxs_per_cls[lab].append(i)

train_idxs, val_idxs, test_idxs = [], [], []
for lab, idxs in idxs_per_cls.items():
    # shuffle per-class indices reproducibly
    torch.manual_seed(SEED + lab)
    idxs = idxs.copy()
    random_state = torch.randperm(len(idxs)).tolist()
    idxs = [idxs[i] for i in random_state]

    n  = len(idxs)
    nt = max(1, int(n * TEST_FRAC))
    test_idxs .extend(idxs[:nt])
    rem        = idxs[nt:]
    nv         = int(len(rem) * TRAIN_FRAC)
    train_idxs.extend(rem[:nv])
    val_idxs  .extend(rem[nv:])

print(f"Splits → train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")

# ─── REMAP TO 3 CLASSES ────────────────────────────────────────────────────────
label_map   = {0:0, 1:1, 2:1, 3:2}  # merge Anxiety+Fear
class_names = ["Aggressiveness", "Anxiety/Fear", "Neutral"]
NUM_CLASSES = len(class_names)

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ─── DATASET WRAPPER ──────────────────────────────────────────────────────────
class RemapDataset(Dataset):
    def __init__(self, samples, indices, transform):
        self.sub = [samples[i] for i in indices]
        self.tf  = transform

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, i):
        path, orig_lab = self.sub[i]
        img = default_loader(path).convert("RGB")
        if self.tf:
            img = self.tf(img)
        return img, label_map[orig_lab]

# build datasets + dataloaders
train_ds = RemapDataset(all_samples, train_idxs, train_tf)
val_ds   = RemapDataset(all_samples, val_idxs,   eval_tf)
test_ds  = RemapDataset(all_samples, test_idxs,  eval_tf)

train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, seed=SEED)
val_loader   = make_dataloader(val_ds,   BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, seed=SEED)
test_loader  = make_dataloader(test_ds,  BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, seed=SEED)

# ─── MODEL & TRAINING ─────────────────────────────────────────────────────────
print("\n=== MobileNetV2 fine-tune (3-class) ===")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
in_ch  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_ch, NUM_CLASSES)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# train + validate with checkpointing & early stopping
history = train_validate(
    model, optimizer, criterion,
    epochs=EPOCHS,
    name="MobileNetV2-3cls",
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    class_names=class_names,
    checkpoint_dir=OUTPUT_DIR,
    early_stopping_patience=10
)

# save curves
save_curve(history['train_acc'], history['val_acc'],
           "MobileNetV2-3cls", "Acc", OUTPUT_DIR)
save_curve(history['train_loss'], history['val_loss'],
           "MobileNetV2-3cls", "Loss", OUTPUT_DIR)

# final test + metrics
run_model(
    name="MobileNetV2-3cls",
    model=model,
    history=history,
    test_loader=test_loader,
    class_names=class_names,
    output_dir=OUTPUT_DIR,
    device=device
)

# EigenCAM overlays
generate_eigencam_overlays(
    model, test_ds, class_names,
    "MobileNetV2-3cls", device,
    OUTPUT_DIR
)

print("\n✅ Done. Outputs in", OUTPUT_DIR)
