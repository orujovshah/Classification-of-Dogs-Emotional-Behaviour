import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from helpers import (
    set_seed,
    make_dataloader,
    infer_flatten_size,
    train_validate,
    save_curve,
    run_model,
    generate_eigencam_overlays
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR         = "./dataset"
IMG_SIZE         = 224
BATCH_SIZE       = 64
NUM_WORKERS      = 4
EPOCHS_HEAD      = 25
EPOCHS_RES_FINE  = 25
EPOCHS_VGG_ONFLY = 50
EPOCHS_MOB       = 50
TEST_FRAC        = 0.15
TRAIN_FRAC       = 0.70
OUTPUT_DIR       = "./outputs_paperfaithful"
SEED             = 42
EARLY_STOPPING   = 10  # patience in epochs

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
set_seed(SEED)

# ─── DATA SPLIT ───────────────────────────────────────────────────────────────
full_ds     = ImageFolder(BASE_DIR, transform=None)
class_names = full_ds.classes
all_samples = full_ds.samples
all_labels  = [lab for (_p, lab) in all_samples]

from collections import defaultdict
idxs_per_cls = defaultdict(list)
for i, lab in enumerate(all_labels):
    idxs_per_cls[lab].append(i)

train_idxs, val_idxs, test_idxs = [], [], []
for lab, idxs in idxs_per_cls.items():
    random.shuffle(idxs)
    nt = max(1, int(len(idxs) * TEST_FRAC))
    test_idxs .extend(idxs[:nt])
    rem        = idxs[nt:]
    nv         = int(len(rem) * TRAIN_FRAC)
    train_idxs .extend(rem[:nv])
    val_idxs   .extend(rem[nv:])

print(f"Splits → train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")

# ─── TRANSFORMS & DATALOADERS ─────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

class FourDataset(torch.utils.data.Dataset):
    def __init__(self, samples, indices, transform):
        self.sub = [samples[i] for i in indices]
        self.tf  = transform

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, i):
        p, lab = self.sub[i]
        img    = default_loader(p).convert("RGB")
        return (self.tf(img), lab) if self.tf else (img, lab)

train_ds = FourDataset(all_samples, train_idxs, train_tf)
val_ds   = FourDataset(all_samples, val_idxs,   eval_tf)
test_ds  = FourDataset(all_samples, test_idxs,  eval_tf)

train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, seed=SEED)
val_loader   = make_dataloader(val_ds,   BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, seed=SEED)
test_loader  = make_dataloader(test_ds,  BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, seed=SEED)

# ─── 1) ResNet50 Two‐Stage Fine‐Tuning ─────────────────────────────────────────
set_seed(SEED)
res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
for p in res.parameters():
    p.requires_grad = False
for p in res.layer4.parameters():
    p.requires_grad = True
res.fc = nn.Linear(res.fc.in_features, len(class_names)).to(device)

opt   = optim.Adam(filter(lambda p: p.requires_grad, res.parameters()), lr=1e-4)
crit  = nn.CrossEntropyLoss()
res_dir = os.path.join(OUTPUT_DIR, "ResNet50")
os.makedirs(res_dir, exist_ok=True)

# Stage 1: head
hist1 = train_validate(
    res, opt, crit, EPOCHS_HEAD, "ResNet-Head",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=res_dir,
    early_stopping_patience=EARLY_STOPPING
)

# Stage 2: fine-tune layer3+4
for p in res.layer3.parameters():
    p.requires_grad = True
opt = optim.Adam(filter(lambda p: p.requires_grad, res.parameters()), lr=1e-5)

hist2 = train_validate(
    res, opt, crit, EPOCHS_RES_FINE, "ResNet-FT",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=res_dir,
    early_stopping_patience=EARLY_STOPPING
)

# Combine histories
history_resnet = {
    'train_acc':  hist1['train_acc']  + hist2['train_acc'],
    'val_acc':    hist1['val_acc']    + hist2['val_acc'],
    'train_loss': hist1['train_loss'] + hist2['train_loss'],
    'val_loss':   hist1['val_loss']   + hist2['val_loss'],
}

# Curves, evaluation, CAM
save_curve(history_resnet['train_acc'], history_resnet['val_acc'],
           "ResNet50", "Acc", res_dir)
save_curve(history_resnet['train_loss'], history_resnet['val_loss'],
           "ResNet50", "Loss", res_dir)

run_model("ResNet50", res, history_resnet, test_loader,
          class_names, res_dir, device)

generate_eigencam_overlays(
    res, test_ds, class_names,
    "ResNet50", device, res_dir
)


# ─── 2) VGG16 On-The-Fly ──────────────────────────────────────────────────────
set_seed(SEED)
vgg_base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
for p in vgg_base.parameters():
    p.requires_grad = False

flat_dim = infer_flatten_size(vgg_base,
                              input_shape=(1,3,IMG_SIZE,IMG_SIZE),
                              device=device)

head = nn.Sequential(
    nn.Linear(flat_dim,256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256,256),     nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256,len(class_names))
).to(device)

class VGGModel(nn.Module):
    def __init__(self, features, classifier):
        super().__init__()
        self.features   = features
        self.flatten    = nn.Flatten()
        self.classifier = classifier
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

vgg_mod = VGGModel(vgg_base, head).to(device)
opt, crit = optim.Adam(vgg_mod.parameters(), lr=1e-4), nn.CrossEntropyLoss()
vgg_dir = os.path.join(OUTPUT_DIR, "VGG16-OnFly")
os.makedirs(vgg_dir, exist_ok=True)

hist_vgg = train_validate(
    vgg_mod, opt, crit, EPOCHS_VGG_ONFLY, "VGG16-OnFly",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=vgg_dir,
    early_stopping_patience=EARLY_STOPPING
)

save_curve(hist_vgg['train_acc'], hist_vgg['val_acc'],
           "VGG16-OnFly", "Acc", vgg_dir)
save_curve(hist_vgg['train_loss'], hist_vgg['val_loss'],
           "VGG16-OnFly", "Loss", vgg_dir)

run_model("VGG16-OnFly", vgg_mod, hist_vgg, test_loader,
          class_names, vgg_dir, device)

generate_eigencam_overlays(
    vgg_mod, test_ds, class_names,
    "VGG16-OnFly", device, vgg_dir
)
torch.cuda.empty_cache()


# ─── 3) MobileNetV2 Fine-Tuning ───────────────────────────────────────────────
set_seed(SEED)
mob = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
mob.classifier[1] = nn.Linear(mob.classifier[1].in_features,
                              len(class_names)).to(device)

opt, crit = optim.Adam(mob.parameters(), lr=1e-4), nn.CrossEntropyLoss()
mob_dir = os.path.join(OUTPUT_DIR, "MobileNetV2")
os.makedirs(mob_dir, exist_ok=True)

hist_mob = train_validate(
    mob, opt, crit, EPOCHS_MOB, "MobileNetV2",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=mob_dir,
    early_stopping_patience=EARLY_STOPPING
)

save_curve(hist_mob['train_acc'], hist_mob['val_acc'],
           "MobileNetV2", "Acc", mob_dir)
save_curve(hist_mob['train_loss'], hist_mob['val_loss'],
           "MobileNetV2", "Loss", mob_dir)

run_model("MobileNetV2", mob, hist_mob, test_loader,
          class_names, mob_dir, device)

generate_eigencam_overlays(
    mob, test_ds, class_names,
    "MobileNetV2", device, mob_dir
)

print("\n✅ All done. Outputs in", OUTPUT_DIR)
