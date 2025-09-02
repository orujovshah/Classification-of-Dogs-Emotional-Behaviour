import os, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

from helpers import (
    set_seed, make_dataloader, train_validate, save_curve,
    run_model, generate_eigencam_overlays
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
COMBINED_ROOT   = "./Combined4"
IMG_SIZE        = 224
BATCH_SIZE      = 16
NUM_WORKERS     = 4
TEST_FRAC       = 0.10     # test = 10%
TRAIN_FRAC      = 0.80     # of the remaining 90%, train = 80% (val = 20%)
STAGE1_EPOCHS   = 10
STAGE2_EPOCHS   = 10
STAGE3_EPOCHS   = 10
OUTPUT_DIR      = "./outputs_experiment_combined_resnet50_coarse"
SEED            = 42

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── TRANSFORMS ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─── SPLITS ────────────────────────────────────────────────────────────────────
def make_splits(root, tf_train, tf_eval, stratify_subfolder=True):
    """Create 72/18/10 train/val/test splits (stratified)."""
    raw = ImageFolder(root, transform=None)
    samples = raw.samples
    idxs = list(range(len(samples)))

    if stratify_subfolder:
        # stratify by top-level macro-class folder name
        stratify_labels = [os.path.normpath(p).split(os.sep)[-2] for p,_ in samples]
    else:
        stratify_labels = [raw.targets[i] for i in idxs]

    tv, test = train_test_split(
        idxs, test_size=TEST_FRAC, stratify=stratify_labels, random_state=SEED
    )
    tv_labels = [stratify_labels[i] for i in tv]
    train, val = train_test_split(
        tv, test_size=(1-TRAIN_FRAC), stratify=tv_labels, random_state=SEED
    )

    ds_train = Subset(ImageFolder(root, transform=tf_train), train)
    ds_val   = Subset(ImageFolder(root, transform=tf_eval),  val)
    ds_test  = Subset(ImageFolder(root, transform=tf_eval),  test)

    dl_train = make_dataloader(ds_train, BATCH_SIZE, True,  NUM_WORKERS, SEED)
    dl_val   = make_dataloader(ds_val,   BATCH_SIZE, False, NUM_WORKERS, SEED)
    dl_test  = make_dataloader(ds_test,  BATCH_SIZE, False, NUM_WORKERS, SEED)
    return dl_train, dl_val, dl_test, raw.classes

# ─── MODEL FACTORY ─────────────────────────────────────────────────────────────
def make_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    # Expose a 'features' handle for Eigen-CAM (expects a list of target layers)
    object.__setattr__(model, 'features', nn.ModuleList([model.layer4]))
    return model.to(device)

# ─── STAGE 1–3: COARSE 4-WAY ONLY ─────────────────────────────────────────────
train_coarse, val_coarse, test_coarse, coarse_classes = make_splits(
    COMBINED_ROOT, train_tf, eval_tf, stratify_subfolder=True
)

model = make_resnet50(len(coarse_classes))
crit  = nn.CrossEntropyLoss()

# Stage 1: head-only (layer4 + fc)
for p in model.parameters(): p.requires_grad = False
for p in model.layer4.parameters(): p.requires_grad = True
for p in model.fc.parameters():     p.requires_grad = True
opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.1)
hist1 = train_validate(model, opt1, crit, STAGE1_EPOCHS, "Coarse-Stage1",
                       train_coarse, val_coarse, device, coarse_classes,
                       checkpoint_dir=os.path.join(OUTPUT_DIR, "Coarse"))

# Stage 2: unfreeze layer3 (as “last major block”), cosine LR
for p in model.layer3.parameters(): p.requires_grad = True
opt2   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.1)
sched2 = CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)
hist2 = {k:[] for k in ['train_acc','val_acc','train_loss','val_loss']}
for ep in range(1, STAGE2_EPOCHS+1):
    h = train_validate(model, opt2, crit, 1, f"Coarse-Stage2-Ep{ep}",
                       train_coarse, val_coarse, device, coarse_classes)
    for k in hist2: hist2[k].append(h[k][0])
    sched2.step()

# Stage 3: full fine-tune
for p in model.parameters(): p.requires_grad = True
opt3   = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
sched3 = CosineAnnealingLR(opt3, T_max=STAGE3_EPOCHS)
hist3 = {k:[] for k in ['train_acc','val_acc','train_loss','val_loss']}
for ep in range(1, STAGE3_EPOCHS+1):
    h = train_validate(model, opt3, crit, 1, f"Coarse-Stage3-Ep{ep}",
                       train_coarse, val_coarse, device, coarse_classes)
    for k in hist3: hist3[k].append(h[k][0])
    sched3.step()

# ─── SAVE CURVES + EVAL + EIGEN-CAM ───────────────────────────────────────────
hist = {
    'train_acc':  hist1['train_acc']  + hist2['train_acc']  + hist3['train_acc'],
    'val_acc':    hist1['val_acc']    + hist2['val_acc']    + hist3['val_acc'],
    'train_loss': hist1['train_loss'] + hist2['train_loss'] + hist3['train_loss'],
    'val_loss':   hist1['val_loss']   + hist2['val_loss']   + hist3['val_loss'],
}
coarse_dir = os.path.join(OUTPUT_DIR, "Coarse")
os.makedirs(coarse_dir, exist_ok=True)

save_curve(hist['train_acc'],  hist['val_acc'],  "Coarse", "Acc",  coarse_dir)
save_curve(hist['train_loss'], hist['val_loss'], "Coarse", "Loss", coarse_dir)

# test evaluation (confusion, report, metrics saved by run_model)
run_model("Coarse", model, hist, test_coarse, coarse_classes, coarse_dir, device)

# Eigen-CAM overlays for a sample of test images
generate_eigencam_overlays(model, test_coarse.dataset, coarse_classes, "Coarse", device, coarse_dir)

print("✅ Coarse-only experiment complete. Outputs in", OUTPUT_DIR)
