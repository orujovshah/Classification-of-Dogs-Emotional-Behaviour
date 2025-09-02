import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import timm

from helpers import (
    set_seed, make_dataloader, train_validate, save_curve,
    run_model, generate_eigencam_overlays
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
COMBINED_ROOT   = "./Combined4"
IMG_SIZE        = 224
BATCH_SIZE      = 8
NUM_WORKERS     = 4
TEST_FRAC       = 0.10
TRAIN_FRAC      = 0.80
STAGE1_EPOCHS   = 10
STAGE2_EPOCHS   = 10
STAGE3_EPOCHS   = 10
OUTPUT_DIR      = "./outputs_experiment_combined_efficientnet_coarse"
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

# ─── SPLITS (72/18/10 stratified by top-level folder) ─────────────────────────
def make_splits(root, tf_train, tf_eval, stratify_subfolder=True):
    raw = ImageFolder(root, transform=None)
    samples = raw.samples
    idxs = list(range(len(samples)))

    if stratify_subfolder:
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
def make_effnetv2m(num_classes):
    model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True)
    # Expose features for Eigen-CAM (timm EfficientNetV2 has .blocks)
    object.__setattr__(model, 'features', model.blocks)
    # Replace classifier head
    in_feats = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_feats, num_classes)
    )
    return model.to(device)

# ─── STAGE 1–3: COARSE 4-WAY ONLY ─────────────────────────────────────────────
train_coarse, val_coarse, test_coarse, coarse_classes = make_splits(
    COMBINED_ROOT, train_tf, eval_tf, stratify_subfolder=True
)

model = make_effnetv2m(len(coarse_classes))
crit  = nn.CrossEntropyLoss()

# Stage 1: head-only
for p in model.parameters(): p.requires_grad = False
for p in model.classifier.parameters(): p.requires_grad = True
opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.1)
h1 = train_validate(model, opt1, crit, STAGE1_EPOCHS, "Coarse-Stage1",
                    train_coarse, val_coarse, device, coarse_classes,
                    checkpoint_dir=os.path.join(OUTPUT_DIR, "Coarse"))

# Stage 2: unfreeze last 3 blocks
for blk in model.blocks[-3:]:
    for p in blk.parameters():
        p.requires_grad = True
opt2   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.1)
sched2 = CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)
h2 = {k:[] for k in ['train_acc','val_acc','train_loss','val_loss']}
for ep in range(1, STAGE2_EPOCHS+1):
    hh = train_validate(model, opt2, crit, 1, f"Coarse-Stage2-Ep{ep}",
                        train_coarse, val_coarse, device, coarse_classes)
    for k in h2: h2[k].append(hh[k][0])
    sched2.step()

# Stage 3: full fine-tune
for p in model.parameters(): p.requires_grad = True
opt3   = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
sched3 = CosineAnnealingLR(opt3, T_max=STAGE3_EPOCHS)
h3 = {k:[] for k in ['train_acc','val_acc','train_loss','val_loss']}
for ep in range(1, STAGE3_EPOCHS+1):
    hh = train_validate(model, opt3, crit, 1, f"Coarse-Stage3-Ep{ep}",
                        train_coarse, val_coarse, device, coarse_classes)
    for k in h3: h3[k].append(hh[k][0])
    sched3.step()

# ─── SAVE CURVES + EVAL + EIGEN-CAM ───────────────────────────────────────────
hist = {
    'train_acc':  h1['train_acc']  + h2['train_acc']  + h3['train_acc'],
    'val_acc':    h1['val_acc']    + h2['val_acc']    + h3['val_acc'],
    'train_loss': h1['train_loss'] + h2['train_loss'] + h3['train_loss'],
    'val_loss':   h1['val_loss']   + h2['val_loss']   + h3['val_loss'],
}
coarse_dir = os.path.join(OUTPUT_DIR, "Coarse")
os.makedirs(coarse_dir, exist_ok=True)

save_curve(hist['train_acc'],  hist['val_acc'],  "Coarse", "Acc",  coarse_dir)
save_curve(hist['train_loss'], hist['val_loss'], "Coarse", "Loss", coarse_dir)

# test evaluation (confusion, report, metrics saved by run_model)
run_model("Coarse", model, hist, test_coarse, coarse_classes, coarse_dir, device)

# Eigen-CAM overlays for a sample of test images
generate_eigencam_overlays(model, test_coarse.dataset, coarse_classes, "Coarse", device, coarse_dir)

print("✅ Coarse-only EfficientNetV2-M experiment complete. Outputs in", OUTPUT_DIR)
