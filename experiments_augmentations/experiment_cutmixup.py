import os
import random
import shutil
from collections import defaultdict
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import models, transforms

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
ORIG_ROOT    = "../dataset"
AUG_ROOT     = "./dataset_experiment_cutmixup"
OUTPUT_DIR   = "./outputs_experiment_cutmixup"
IMG_SIZE     = 224
BATCH_SIZE   = 64
NUM_WORKERS  = 4
SEED         = 42
E1_HEAD      = 25
E1_FINETUNE  = 25
E2_VGG       = 50
E2_MOB       = 50
TEST_FRAC, TRAIN_FRAC = 0.15, 0.70
MU_ALPHA     = 0.4
CM_ALPHA     = 1.0

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

os.makedirs(AUG_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── STEP 1: LOAD & STRATIFIED SPLIT ────────────────────────────────────────────
full_ds = []
class_names = sorted(d for d in os.listdir(ORIG_ROOT)
                     if os.path.isdir(os.path.join(ORIG_ROOT, d)))
for cls_idx, cls in enumerate(class_names):
    for fn in os.listdir(os.path.join(ORIG_ROOT, cls)):
        if fn.lower().endswith(('.png','jpg','jpeg','bmp','gif')):
            full_ds.append((os.path.join(ORIG_ROOT, cls, fn), cls_idx))

idxs_per_cls = defaultdict(list)
for i, (_, lab) in enumerate(full_ds):
    idxs_per_cls[lab].append(i)

train_idxs, val_idxs, test_idxs = [], [], []
for lab, idxs in idxs_per_cls.items():
    random.seed(SEED + lab)
    random.shuffle(idxs)
    n_test = max(1, int(len(idxs) * TEST_FRAC))
    test_idxs.extend(idxs[:n_test])
    rem = idxs[n_test:]
    n_tr = int(len(rem) * TRAIN_FRAC)
    train_idxs.extend(rem[:n_tr])
    val_idxs.extend(rem[n_tr:])

print(f"Splits → train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")

# ─── STEP 2: ON-DISK MIXUP / CUTMIX / MIXCUT AUGMENTATION ─────────────────────
base_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
for cls_idx, cls in enumerate(class_names):
    base_dir = os.path.join(AUG_ROOT, cls)
    for sub in ("Originals", "MixUp", "CutMix", "MixCut"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    paths = [full_ds[i][0] for i in train_idxs if full_ds[i][1] == cls_idx]
    n_half = len(paths) // 2

    # Originals
    for p in paths:
        shutil.copy(p, os.path.join(base_dir, "Originals"))

    # MixUp
    for i, p1 in enumerate(paths[:n_half]):
        p2 = random.choice(paths)
        img1 = base_tf(Image.open(p1).convert("RGB"))
        img2 = base_tf(Image.open(p2).convert("RGB"))
        lam = np.random.beta(MU_ALPHA, MU_ALPHA)
        mixed = (lam * img1 + (1 - lam) * img2).clamp(0, 1)
        T.ToPILImage()(mixed).save(os.path.join(base_dir, "MixUp", f"mixup_{i:04d}.png"))

    # CutMix
    for i, p1 in enumerate(paths[:n_half]):
        p2 = random.choice(paths)
        img1 = base_tf(Image.open(p1).convert("RGB"))
        img2 = base_tf(Image.open(p2).convert("RGB"))
        lam = np.random.beta(CM_ALPHA, CM_ALPHA)
        cut_w = int(IMG_SIZE * np.sqrt(1 - lam))
        cut_h = cut_w
        cx = random.randint(0, IMG_SIZE - cut_w)
        cy = random.randint(0, IMG_SIZE - cut_h)
        img1[:, cy:cy+cut_h, cx:cx+cut_w] = img2[:, cy:cy+cut_h, cx:cx+cut_w]
        mixed = img1.clamp(0, 1)
        T.ToPILImage()(mixed).save(os.path.join(base_dir, "CutMix", f"cutmix_{i:04d}.png"))

    # MixCut
    for i, p1 in enumerate(paths[:n_half]):
        p2, p3 = random.choice(paths), random.choice(paths)
        img1 = base_tf(Image.open(p1).convert("RGB"))
        img2 = base_tf(Image.open(p2).convert("RGB"))
        lam1 = np.random.beta(MU_ALPHA, MU_ALPHA)
        mixed1 = lam1 * img1 + (1 - lam1) * img2
        lam2 = np.random.beta(CM_ALPHA, CM_ALPHA)
        cut_w = int(IMG_SIZE * np.sqrt(1 - lam2))
        cx = random.randint(0, IMG_SIZE - cut_w)
        cy = random.randint(0, IMG_SIZE - cut_w)
        img3 = base_tf(Image.open(p3).convert("RGB"))
        mixed1[:, cy:cy+cut_w, cx:cx+cut_w] = img3[:, cy:cy+cut_w, cx:cx+cut_w]
        T.ToPILImage()(mixed1.clamp(0, 1)).save(os.path.join(base_dir, "MixCut", f"mixcut_{i:04d}.png"))

# collect augmented samples
all_aug = []
for cls_idx, cls in enumerate(class_names):
    for sub in ("Originals", "MixUp", "CutMix", "MixCut"):
        folder = os.path.join(AUG_ROOT, cls, sub)
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.png','jpg','jpeg','bmp','gif')):
                all_aug.append((os.path.join(folder, fn), cls_idx))
train_idxs_aug = list(range(len(all_aug)))
print(f"Augmented train size = {len(train_idxs_aug)}")

# ─── DATASET & DATALOADERS ────────────────────────────────────────────────────
class FourDataset(Dataset):
    def __init__(self, samples, idxs, tf):
        self.sub = [samples[i] for i in idxs]
        self.tf  = tf
    def __len__(self): return len(self.sub)
    def __getitem__(self, i):
        p, lab = self.sub[i]
        img = default_loader(p).convert("RGB")
        return self.tf(img), lab

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_loader_res = make_dataloader(
    FourDataset(all_aug, train_idxs_aug, train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, seed=SEED
)
train_loader_vgg = make_dataloader(
    FourDataset(all_aug, train_idxs_aug, train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, seed=SEED
)
train_loader_mob = make_dataloader(
    FourDataset(all_aug, train_idxs_aug, train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, seed=SEED
)
val_loader = make_dataloader(
    FourDataset(full_ds, val_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, seed=SEED
)
test_loader = make_dataloader(
    FourDataset(full_ds, test_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, seed=SEED
)

# ─── UTILITY FOR EVALUATION ───────────────────────────────────────────────────
def evaluate_model(name, model, history):
    mdir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(mdir, exist_ok=True)
    save_curve(history['train_acc'], history['val_acc'], name, "Acc", mdir)
    save_curve(history['train_loss'], history['val_loss'], name, "Loss", mdir)
    run_model(name, model, history, test_loader, class_names, mdir, device)
    generate_eigencam_overlays(model, test_loader.dataset, class_names, name, device, mdir)
    torch.cuda.empty_cache()

# ─── EXPERIMENT 1: ResNet50 ────────────────────────────────────────────────────
set_seed(SEED)
res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
for p in res.parameters(): p.requires_grad = False
for p in res.layer4.parameters(): p.requires_grad = True
res.fc = nn.Linear(res.fc.in_features, len(class_names)).to(device)

# Stage 1
opt1 = optim.Adam(res.fc.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
hist1 = train_validate(
    res, opt1, crit, E1_HEAD, "ResNet-Head",
    train_loader_res, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "ResNet50")
)

# Stage 2
for p in res.layer3.parameters(): p.requires_grad = True
opt2 = optim.Adam(filter(lambda p: p.requires_grad, res.parameters()), lr=1e-5)
hist2 = train_validate(
    res, opt2, crit, E1_FINETUNE, "ResNet-FT",
    train_loader_res, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "ResNet50")
)

history_res = {
    'train_acc':  hist1['train_acc']  + hist2['train_acc'],
    'val_acc':    hist1['val_acc']    + hist2['val_acc'],
    'train_loss': hist1['train_loss'] + hist2['train_loss'],
    'val_loss':   hist1['val_loss']   + hist2['val_loss'],
}
evaluate_model("ResNet50", res, history_res)

# ─── EXPERIMENT 2: VGG16 ───────────────────────────────────────────────────────
set_seed(SEED)
vgg_base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
for p in vgg_base.parameters(): p.requires_grad = False

flat_dim = infer_flatten_size(vgg_base, input_shape=(1,3,IMG_SIZE,IMG_SIZE), device=device)
head = nn.Sequential(
    nn.Linear(flat_dim, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, len(class_names))
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
opt_vgg = optim.Adam(filter(lambda p: p.requires_grad, vgg_mod.parameters()), lr=1e-4)
hist_vgg = train_validate(
    vgg_mod, opt_vgg, crit, E2_VGG, "VGG16-OnFly",
    train_loader_vgg, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "VGG16-OnFly")
)
evaluate_model("VGG16-OnFly", vgg_mod, hist_vgg)

# ─── EXPERIMENT 3: MobileNetV2 ─────────────────────────────────────────────────
set_seed(SEED)
mob = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
mob.classifier[1] = nn.Linear(mob.classifier[1].in_features, len(class_names)).to(device)

opt_mob = optim.Adam(mob.parameters(), lr=1e-4)
hist_mob = train_validate(
    mob, opt_mob, crit, E2_MOB, "MobileNetV2",
    train_loader_mob, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "MobileNetV2")
)
evaluate_model("MobileNetV2", mob, hist_mob)

print("\n✅ All done. Outputs in", OUTPUT_DIR)
