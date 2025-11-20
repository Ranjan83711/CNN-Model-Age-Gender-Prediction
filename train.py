# src/train.py
"""
Train script for UTKFace multi-task (age regression + gender classification).

Example:
    python src/train.py --data_dir ../data/UTKFace --epochs 10 --batch_size 32
"""
import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import UTKFaceDataset
from src.models.multitask_resnet import MultiTaskResNet
from src.utils import save_checkpoint, load_checkpoint

from sklearn.model_selection import train_test_split

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_num_workers(requested: int):
    if os.name == 'nt':
        return 0
    return requested

def evaluate(model, dataloader, device, criterion_age, criterion_gender):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_age_abs_error = 0.0
    correct_gender = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            ages = batch['age'].float().to(device)
            genders = batch['gender'].long().to(device)

            age_pred, gender_pred = model(imgs)
            loss_age = criterion_age(age_pred, ages)
            loss_gender = criterion_gender(gender_pred, genders)
            loss = loss_age + 0.5 * loss_gender

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            total_age_abs_error += torch.sum(torch.abs(age_pred - ages)).item()
            preds = torch.argmax(gender_pred, dim=1)
            correct_gender += torch.sum(preds == genders).item()

    avg_loss = total_loss / total_samples
    mae_age = total_age_abs_error / total_samples
    gender_acc = correct_gender / total_samples
    return avg_loss, mae_age, gender_acc

def _parse_gender_from_filename(fp: str):
    """
    Try to parse UTKFace filename: 'age_gender_race_date.jpg' -> gender as int
    Returns None on failure.
    """
    try:
        base = os.path.basename(fp)
        name = os.path.splitext(base)[0]
        parts = name.split('_')
        if len(parts) >= 2:
            return int(parts[1])
    except Exception:
        pass
    return None

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = UTKFaceDataset(args.data_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {args.data_dir} - check path and dataset.")

    # ----------------------------
    # Subsampling logic (minimal & reversible)
    # ----------------------------
    # If max_samples == -1 -> use all
    if args.max_samples != -1:
        n_total = len(dataset)
        if args.max_samples is not None and args.max_samples < n_total:
            rng = random.Random(args.seed)
            # Balanced sampling by gender
            if args.balanced_by_gender:
                print(f"[INFO] Building balanced subset by gender (max_per_gender={args.max_per_gender})")
                # Try to read file paths quickly if dataset exposes them
                files_attr = getattr(dataset, 'files', None) or getattr(dataset, 'image_paths', None) or getattr(dataset, 'imgs', None)
                idxs_by_gender = defaultdict(list)

                if files_attr is not None:
                    # files_attr may be list of filepaths or (path, label) tuples; normalize
                    for idx, fp in enumerate(files_attr):
                        # if tuple like (fp, label) handle it
                        if isinstance(fp, (list, tuple)) and len(fp) > 0:
                            fp = fp[0]
                        g = _parse_gender_from_filename(fp)
                        if g is None:
                            continue
                        idxs_by_gender[g].append(idx)
                else:
                    # fallback: iterate dataset once and inspect label (slower)
                    for idx in range(len(dataset)):
                        try:
                            item = dataset[idx]
                            # assume item contains 'gender' in dict or tuple
                            if isinstance(item, dict):
                                g = item.get('gender', None)
                            else:
                                # maybe returns (img, gender, age)
                                if len(item) >= 2:
                                    g = item[1]
                                else:
                                    g = None
                            if g is None:
                                continue
                            # if tensor -> extract
                            if hasattr(g, 'item'):
                                g = int(g.item())
                            else:
                                g = int(g)
                            idxs_by_gender[g].append(idx)
                        except Exception:
                            continue

                selected = []
                for g, idx_list in idxs_by_gender.items():
                    rng.shuffle(idx_list)
                    selected.extend(idx_list[:min(len(idx_list), args.max_per_gender)])
                rng.shuffle(selected)
                if len(selected) == 0:
                    raise RuntimeError("Balanced sampling resulted in 0 selected indices. Check dataset genders.")
                print(f"[INFO] Selected {len(selected)} samples (balanced).")
                # use Subset for later splitting
                dataset = Subset(dataset, selected)
            else:
                # Simple random subset
                print(f"[INFO] Sampling a random subset of {args.max_samples} / {n_total} total samples")
                indices = list(range(n_total))
                rng = random.Random(args.seed)
                rng.shuffle(indices)
                chosen_idx = indices[:args.max_samples]
                dataset = Subset(dataset, chosen_idx)
                print(f"[INFO] Selected {len(chosen_idx)} samples (random).")
        else:
            print("[INFO] max_samples >= dataset size — using full dataset")
    else:
        print("[INFO] max_samples set to -1 -> using full dataset")

    # Save selected indices for reproducibility (best-effort)
    try:
        if isinstance(dataset, Subset):
            with open("selected_subset_indices.json", "w") as f:
                json.dump(dataset.indices, f)
    except Exception:
        pass
    # ----------------------------
    # End subsampling
    # ----------------------------

    # build indices and splits
    idx = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(idx, test_size=(args.val_frac + args.test_frac), random_state=args.seed)
    val_size = int(len(idx) * args.val_frac)
    val_idx = temp_idx[:val_size]
    test_idx = temp_idx[val_size:]

    train_ds = Subset(dataset, train_idx) if not isinstance(dataset, Subset) else Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if not isinstance(dataset, Subset) else Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx) if not isinstance(dataset, Subset) else Subset(dataset, test_idx)

    num_workers = get_num_workers(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[INFO] Sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    model = MultiTaskResNet(pretrained=args.pretrained).to(device)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.isfile(args.checkpoint):
        ckpt = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', best_val_loss)
        print(f"[INFO] Resumed from checkpoint at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_age = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=120)
        for batch in pbar:
            imgs = batch['image'].to(device)
            ages = batch['age'].float().to(device)
            genders = batch['gender'].long().to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.use_amp)):
                age_pred, gender_pred = model(imgs)
                loss_age = criterion_age(age_pred, ages)
                loss_gender = criterion_gender(gender_pred, genders)
                loss = loss_age + args.gender_loss_weight * loss_gender

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        print(f"[INFO] Epoch {epoch+1} train loss: {epoch_train_loss:.4f}")

        val_loss, val_mae, val_acc = evaluate(model, val_loader, device, criterion_age, criterion_gender)
        print(f"[INFO] Epoch {epoch+1} val loss: {val_loss:.4f} | val MAE(age): {val_mae:.4f} | val acc(gender): {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss
            }, args.checkpoint)
            print("[INFO] Saved best checkpoint.")

    # final test evaluation
    test_loss, test_mae, test_acc = evaluate(model, test_loader, device, criterion_age, criterion_gender)
    print(f"[RESULTS] Test loss: {test_loss:.4f} | Test MAE(age): {test_mae:.4f} | Test acc(gender): {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/UTKFace')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone weights')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--load_optimizer', action='store_true', help='Load optimizer state when resuming')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--gender_loss_weight', type=float, default=0.5)

    # NEW options for sampling (small dataset runs)
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Total images to sample for training. Set -1 to use all data.')
    parser.add_argument('--balanced_by_gender', action='store_true',
                        help='If set, select up to --max_per_gender samples per gender (balanced).')
    parser.add_argument('--max_per_gender', type=int, default=1000,
                        help='Max samples per gender when --balanced_by_gender is used.')

    args = parser.parse_args()
    Path(os.path.dirname(args.checkpoint) or ".").mkdir(parents=True, exist_ok=True)
    train(args)
