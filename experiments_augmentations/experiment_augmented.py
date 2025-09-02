import os
import random
import shutil
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import models, transforms
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
ORIG_ROOT      = "../dataset"
AUG_ROOT       = "./dataset_experiment_augmented"
OUTPUT_DIR     = "./outputs_experiment_augmented"
IMG_SIZE       = 224
BATCH_SIZE     = 64
NUM_WORKERS    = 4
SEED           = 42
TEST_FRAC      = 0.15
TRAIN_FRAC     = 0.70

E1_HEAD        = 25
E1_FINETUNE    = 25
E2_VGG         = 50
E2_MOB         = 50

os.makedirs(AUG_ROOT,    exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── STEP 1: LOAD & STRATIFIED SPLIT ───────────────────────────────────────────
full_ds_orig = []
class_names  = sorted(d for d in os.listdir(ORIG_ROOT)
                      if os.path.isdir(os.path.join(ORIG_ROOT, d)))
for idx, cls in enumerate(class_names):
    for fn in os.listdir(os.path.join(ORIG_ROOT, cls)):
        if fn.lower().endswith(('.png','jpg','jpeg','bmp','gif')):
            full_ds_orig.append((os.path.join(ORIG_ROOT, cls, fn), idx))

idxs_per_cls = defaultdict(list)
for i, (_p, lab) in enumerate(full_ds_orig):
    idxs_per_cls[lab].append(i)

train_idxs, val_idxs, test_idxs = [], [], []
for lab, idxs in idxs_per_cls.items():
    random.seed(SEED + lab)
    random.shuffle(idxs)
    nt = max(1, int(len(idxs) * TEST_FRAC))
    test_idxs.extend(idxs[:nt])
    rem = idxs[nt:]
    nv  = int(len(rem) * TRAIN_FRAC)
    train_idxs.extend(rem[:nv])
    val_idxs.extend(rem[nv:])

print(f"Splits → train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")

# ─── STEP 2: ON-DISK AUGMENTATION ──────────────────────────────────────────────
for cls_idx, cls in enumerate(class_names):
    base = os.path.join(AUG_ROOT, cls)
    for sub in ("Originals","Erasing","Mosaic","Geometric"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    paths = [full_ds_orig[i][0] for i in train_idxs if full_ds_orig[i][1] == cls_idx]

    # Originals
    for p in paths:
        shutil.copy(p, os.path.join(base, "Originals"))

    # Erasing
    erase_tf = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.ToTensor(),
        T.RandomErasing(p=1.0, scale=(0.02,0.2), ratio=(0.3,3.3), value='random'),
    ])
    for i, p in enumerate(paths):
        img_t = erase_tf(Image.open(p).convert("RGB"))
        T.ToPILImage()(img_t).save(os.path.join(base,"Erasing", f"erased_{i:03d}.png"))

    # Mosaic
    for i in range(len(paths)//4):
        picks = random.sample(paths, 4)
        tiles = [Image.open(q).convert("RGB").resize((IMG_SIZE//2,IMG_SIZE//2))
                 for q in picks]
        mos = Image.new("RGB",(IMG_SIZE,IMG_SIZE))
        mos.paste(tiles[0], (0,0));         mos.paste(tiles[1], (IMG_SIZE//2,0))
        mos.paste(tiles[2], (0,IMG_SIZE//2)); mos.paste(tiles[3], (IMG_SIZE//2,IMG_SIZE//2))
        mos.save(os.path.join(base,"Mosaic", f"mosaic_{i:03d}.jpg"))

    # Geometric
    geo_tf = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
        T.RandomAffine(degrees=15, translate=(0.1,0.1), shear=10),
        T.ToTensor()
    ])
    for i, p in enumerate(paths):
        img_t = geo_tf(Image.open(p).convert("RGB"))
        T.ToPILImage()(img_t).save(os.path.join(base,"Geometric", f"geo_{i:03d}.png"))

# collect augmented samples
all_aug = []
for cls_idx, cls in enumerate(class_names):
    base = os.path.join(AUG_ROOT, cls)
    for sub in ("Originals","Erasing","Mosaic","Geometric"):
        for fn in os.listdir(os.path.join(base, sub)):
            if fn.lower().endswith(('.png','jpg','jpeg','bmp','gif')):
                all_aug.append((os.path.join(base, sub, fn), cls_idx))
train_idxs_aug = list(range(len(all_aug)))
print(f"Augmented train size = {len(train_idxs_aug)}")

# ─── STEP 3: TRANSFORMS & DATALOADERS ──────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class FourDataset(Dataset):
    def __init__(self, samples, idxs, tf):
        self.sub = [samples[i] for i in idxs]
        self.tf  = tf
    def __len__(self):
        return len(self.sub)
    def __getitem__(self, i):
        p, lab = self.sub[i]
        img = default_loader(p).convert("RGB")
        return (self.tf(img), lab) if self.tf else (img, lab)

train_loader = make_dataloader(
    FourDataset(all_aug, train_idxs_aug, train_tf),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, seed=SEED
)
val_loader = make_dataloader(
    FourDataset(full_ds_orig, val_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, seed=SEED
)
test_loader = make_dataloader(
    FourDataset(full_ds_orig, test_idxs, eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, seed=SEED
)

# ─── MODEL EXPERIMENT: AUGMENTED TRAIN ─────────────────────────────────────────

# Utility to evaluate & save per‐model
def evaluate_model(name, model, history):
    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    save_curve(history['train_acc'], history['val_acc'], name, "Acc", model_dir)
    save_curve(history['train_loss'], history['val_loss'], name, "Loss", model_dir)
    run_model(name, model, history, test_loader, class_names, model_dir, device)
    generate_eigencam_overlays(model, test_loader.dataset, class_names, name, device, model_dir)

# ─── A) ResNet50 Two‐Stage ────────────────────────────────────────────────────
set_seed(SEED)
res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
for p in res.parameters(): p.requires_grad = False
for p in res.layer4.parameters(): p.requires_grad = True
res.fc = nn.Linear(res.fc.in_features, len(class_names)).to(device)

# Stage 1: head
opt = torch.optim.Adam(res.fc.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
hist1 = train_validate(
    res, opt, crit, E1_HEAD, "WithAug-ResNet-Head",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "ResNet50")
)

# Stage 2: fine‐tune layer3+4
for p in res.layer3.parameters(): p.requires_grad = True
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, res.parameters()), lr=1e-5)
hist2 = train_validate(
    res, opt, crit, E1_FINETUNE, "WithAug-ResNet-FT",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "ResNet50")
)

history_res = {
    'train_acc':  hist1['train_acc']  + hist2['train_acc'],
    'val_acc':    hist1['val_acc']    + hist2['val_acc'],
    'train_loss': hist1['train_loss'] + hist2['train_loss'],
    'val_loss':   hist1['val_loss']   + hist2['val_loss'],
}
evaluate_model("WithAug-ResNet50", res, history_res)

# ─── B) VGG16 On‐The‐Fly ───────────────────────────────────────────────────────
set_seed(SEED)
vgg_base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
for p in vgg_base.parameters(): p.requires_grad = False

flat_dim = infer_flatten_size(vgg_base, input_shape=(1,3,IMG_SIZE,IMG_SIZE), device=device)
head = nn.Sequential(
    nn.Linear(flat_dim, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 256),     nn.ReLU(), nn.Dropout(0.5),
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
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg_mod.parameters()), lr=1e-4)
crit = nn.CrossEntropyLoss()
hist_vgg = train_validate(
    vgg_mod, opt, crit, E2_VGG, "WithAug-VGG16-OnFly",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "VGG16-OnFly")
)
evaluate_model("WithAug-VGG16-OnFly", vgg_mod, hist_vgg)

# ─── C) MobileNetV2 Fine-Tuning ────────────────────────────────────────────────
set_seed(SEED)
mob = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
mob.classifier[1] = nn.Linear(mob.classifier[1].in_features, len(class_names)).to(device)

opt = torch.optim.Adam(mob.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
hist_mob = train_validate(
    mob, opt, crit, E2_MOB, "WithAug-MobileNetV2",
    train_loader, val_loader, device, class_names,
    checkpoint_dir=os.path.join(OUTPUT_DIR, "MobileNetV2")
)
evaluate_model("WithAug-MobileNetV2", mob, hist_mob)

print("\n✅ Done. Outputs in", OUTPUT_DIR)
