import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Reproducibility utilities ────────────────────────────────────────────────
def set_seed(seed: int):
    """Seed Python, NumPy, Torch and CuDNN for full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def get_worker_init_fn(seed: int):
    """Returns a worker_init_fn for DataLoader that seeds each worker."""
    def worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return worker_init_fn

def make_dataloader(dataset, batch_size, shuffle, num_workers, seed):
    """
    Build a DataLoader with a seeded torch.Generator and worker_init_fn.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=get_worker_init_fn(seed),
    )

# ── Training + Validation ────────────────────────────────────────────────────
def train_validate(
    model, optimizer, criterion, epochs, name,
    train_loader, val_loader, device, class_names,
    checkpoint_dir=None,            # e.g. "./checkpoints"
    early_stopping_patience=None    # e.g. 5 epochs
):
    """
    Trains for `epochs`, optionally checkpointing best-val models and early stopping.
    Returns history dict with 'train_acc', 'val_acc', 'train_loss', 'val_loss'.
    """
    os.makedirs(checkpoint_dir or "./", exist_ok=True)
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for e in range(1, epochs+1):
        # ——— Train —————————————————————————————————————————————————————
        model.train()
        correct, total, run_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            preds = out.argmax(1)
            correct   += (preds == yb).sum().item()
            total     += yb.size(0)
            run_loss  += loss.item()
        tr_acc = correct/total
        tr_loss = run_loss/len(train_loader)

        # ——— Validate ———————————————————————————————————————————————————
        model.eval()
        correct, total, run_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                preds = out.argmax(1)
                correct   += (preds == yb).sum().item()
                total     += yb.size(0)
                run_loss  += loss.item()
        vl_acc = correct/total
        vl_loss = run_loss/len(val_loader)

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)

        print(f"[{name}] Ep{e:02d}/{epochs}  TrainAcc {tr_acc:.4f}  ValAcc {vl_acc:.4f}")

        # ——— Checkpoint + Early-Stopping ——————————————————————————————
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"{name}_best.pth")
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1

            if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping after {e} epochs (no improvement in {early_stopping_patience} epochs).")
                break

    # ——— Load best weights if checkpointed ———————————————————————————
    if checkpoint_dir:
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{name}_best.pth")))

    return history

# ── Metrics & Curves ─────────────────────────────────────────────────────────
def save_curve(vals1, vals2, name, what, output_dir):
    """Plot and save training vs validation curves."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(vals1, label=f"Train {what}")
    plt.plot(vals2, label=f"Val   {what}")
    plt.xlabel("Epochs"); plt.ylabel(what)
    plt.title(f"{name} {what} Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_{what.lower()}_curve.svg"), format='svg')
    plt.close()

def save_confusion(all_y, all_p, name, class_names, output_dir):
    cm = confusion_matrix(all_y, all_p)
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues'); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_title(f"{name} Confusion")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_conf.svg"), format='svg')
    plt.close()

def save_metrics(name, history, all_y, all_p, class_names, output_dir):
    """
    Writes:
     - Overall test accuracy
     - Train/Val loss & acc extrema from `history`
     - Macro & weighted F1
     - Per-class Precision/Recall/F1/Supp/Acc
    """
    rpt = classification_report(all_y, all_p, target_names=class_names,
                                zero_division=0, output_dict=True)
    cm = confusion_matrix(all_y, all_p)
    per_acc = np.diag(cm) / cm.sum(axis=1)

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{name}_metrics.txt")
    with open(fname, "w") as f:
        f.write(f"[{name}] Metrics\n{'='*40}\n\n")
        f.write(f"Test accuracy:        {np.mean(np.array(all_y)==np.array(all_p)):.4f}\n\n")
        f.write(f"Best train  acc:      {max(history['train_acc']):.4f}\n")
        f.write(f"Worst train acc:      {min(history['train_acc']):.4f}\n")
        f.write(f"Best val    acc:      {max(history['val_acc']):.4f}\n")
        f.write(f"Worst val   acc:      {min(history['val_acc']):.4f}\n\n")
        f.write(f"Lowest train  loss:   {min(history['train_loss']):.4f}\n")
        f.write(f"Highest train loss:   {max(history['train_loss']):.4f}\n")
        f.write(f"Lowest val    loss:   {min(history['val_loss']):.4f}\n")
        f.write(f"Highest val   loss:   {max(history['val_loss']):.4f}\n\n")
        # Macro & weighted F1:
        f.write(f"Macro F1:             {rpt['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted F1:          {rpt['weighted avg']['f1-score']:.4f}\n\n")
        # Per-class breakdown
        f.write(f"{'Class':<15}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Supp':>8}{'Acc':>8}\n")
        f.write("-"*55 + "\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls:<15}"
                    f"{rpt[cls]['precision']:8.4f}"
                    f"{rpt[cls]['recall']:8.4f}"
                    f"{rpt[cls]['f1-score']:8.4f}"
                    f"{int(rpt[cls]['support']):8d}"
                    f"{per_acc[i]:8.4f}\n")
    print(f"→ saved metrics to {fname}")

# ── Model evaluation ──────────────────────────────────────────────────────────
def run_model(name, model, history, test_loader, class_names, output_dir, device):
    """
    Evaluates `model` on test_loader, saves confusion & metrics, then deletes model.
    """
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().tolist()
            all_p += preds
            all_y += yb.tolist()

    save_confusion(all_y, all_p, name, class_names, output_dir)
    save_metrics(name, history, all_y, all_p, class_names, output_dir)

    # free GPU memory
    del model
    torch.cuda.empty_cache()

# ── EigenCAM overlays ─────────────────────────────────────────────────────────
def find_last_conv_layer(model):
    """Return the last nn.Conv2d layer in `model`."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model")
    return last_conv

def infer_flatten_size(feature_extractor, input_shape=(1,3,224,224), device='cpu'):
    """
    Runs a dummy input through `feature_extractor` to compute the flattened vector size.
    """
    with torch.no_grad():
        dummy = torch.zeros(input_shape).to(device)
        out   = feature_extractor(dummy)
    return int(np.prod(out.shape[1:]))

def generate_eigencam_overlays(
    model, dataset, class_names,
    model_name, device, output_dir,
    mean=None, std=None
):
    if mean is None: mean = [0.485,0.456,0.406]
    if std  is None: std  = [0.229,0.224,0.225]

    target_layer = find_last_conv_layer(model)
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

    def _build(bucket, kind):
        imgs, titles = [], []
        for cls_idx, entry in bucket.items():
            if entry is None: continue
            idx, pred = entry
            img, label = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            with EigenCAM(model=model, target_layers=[target_layer]) as cam:
                gcam = cam(input_tensor=inp,
                           targets=[ClassifierOutputTarget(pred)])[0]
            arr = (img.cpu().numpy().transpose(1,2,0) * std + mean).clip(0,1)
            overlay = show_cam_on_image(arr, gcam, use_rgb=True)
            imgs.append(overlay)
            titles.append(f"{class_names[label]}\n(pred={class_names[pred]})")

        if not imgs: return
        fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs),4))
        if len(imgs) == 1: axes = [axes]
        for ax, im, tt in zip(axes, imgs, titles):
            ax.imshow(im); ax.set_title(tt, fontsize=9); ax.axis('off')
        plt.tight_layout(pad=1)
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_{kind}_overlays.svg"),
            format='svg', bbox_inches='tight'
        )
        plt.close()

    _build(correct,   "correct")
    _build(incorrect, "incorrect")
