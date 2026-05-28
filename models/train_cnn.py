"""
Tile-based CNN for flood risk prediction -- Group 8
Train on Severn, evaluate on Northumbria.

GPU optimisations applied:
  - AMP (fp16 autocast + GradScaler) for ~2x throughput
  - torch.backends.cudnn.benchmark for faster convolutions
  - Shared-memory tensors + num_workers=4 so CPU data loading
    overlaps with GPU compute
  - Larger batch size (1024) suited to RTX 4070 8 GB VRAM

Run:
    python models/train_cnn.py
"""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from sklearn.metrics import classification_report, cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR   = Path(__file__).parent
REPO_ROOT   = MODEL_DIR.parent
DATA_DIR    = REPO_ROOT / "Data"
CLEAN_DIR   = DATA_DIR / "cleaned"
RESULTS_DIR = MODEL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if USE_AMP:
    torch.backends.cudnn.benchmark = True

# ── Config ─────────────────────────────────────────────────────────────────
PATCH_SIZE   = 31
HALF         = PATCH_SIZE // 2
N_TERRAIN_CH = 6            # dtm_zscore, log_flow_acc, imd, waw, clc, is_waterway
N_CLASSES    = 5
N_TARGETS    = 5

TRAIN_SAMPLES = 1_500_000
TEST_SAMPLES  =   400_000
BATCH_SIZE    = 1024 if DEVICE.type == "cuda" else 256
NUM_WORKERS   = 4    if DEVICE.type == "cuda" else 0
EPOCHS        = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-4

TARGETS    = ["risk_0_2m", "risk_0_3m", "risk_0_6m", "risk_0_9m", "risk_1_2m"]
ERA5_META  = {"pixel_id", "proj_y", "proj_x", "region"}
UNSEEN_CLC = {332}

# Down-weight dominant no-risk class in the loss
LOSS_WEIGHTS = torch.tensor([0.1, 1.0, 2.0, 4.0, 6.0], dtype=torch.float32)


# ── Raster loading ─────────────────────────────────────────────────────────

def load_raster(region: str):
    """
    Load cleaned terrain NetCDF into a shared-memory torch.Tensor
    so DataLoader workers can access it without pickling the array.
    Returns: raster_tensor (C,H,W), valid_tensor (H,W), y0, x0, dy, dx, H, W
    """
    nc = CLEAN_DIR / f"terrain_{region}_clean.nc"
    ds = xr.open_dataset(nc)

    def safe(name, fill=0.0):
        v = ds[name].values.astype(np.float32)
        return np.where(np.isfinite(v), v, fill)

    dtm    = safe("dtm_zscore")
    logacc = safe("log_flow_acc")
    imd    = np.clip(safe("imd"), 0.0, 100.0) / 100.0
    waw    = np.where(np.isfinite(ds["waw"].values),
                      ds["waw"].values, 0.0).astype(np.float32) / 5.0
    clc_raw = ds["clc_type"].values.astype(np.float32)
    clc_raw = np.where(np.isfinite(clc_raw), clc_raw, 0.0)
    clc_raw = np.where(np.isin(clc_raw.astype(int), list(UNSEEN_CLC)), 0.0, clc_raw)
    clc     = clc_raw / 500.0
    isww    = (~np.isnan(ds["rciw"].values)).astype(np.float32)
    valid   = ds["valid_pixel"].values.astype(np.float32)  # float for masking

    y_coords = ds.coords["y"].values
    x_coords = ds.coords["x"].values
    ds.close()

    raster_np = np.stack([dtm, logacc, imd, waw, clc, isww], axis=0)  # (6, H, W)
    H, W = raster_np.shape[1], raster_np.shape[2]

    # Put in shared memory so worker processes can read without copying
    raster_t = torch.from_numpy(raster_np).share_memory_()
    valid_t  = torch.from_numpy(valid).share_memory_()

    y0, dy = float(y_coords[0]), float(y_coords[1] - y_coords[0])
    x0, dx = float(x_coords[0]), float(x_coords[1] - x_coords[0])

    print(f"  Raster {region}: {raster_t.shape}  "
          f"{raster_t.numel()*4/1e9:.2f} GB  valid={int(valid_t.sum()):,}")
    return raster_t, valid_t, y0, x0, dy, dx, H, W


def proj_to_rc(proj_y, proj_x, y0, x0, dy, dx, H, W):
    row = np.round((np.asarray(proj_y, np.float64) - y0) / dy).astype(np.int32)
    col = np.round((np.asarray(proj_x, np.float64) - x0) / dx).astype(np.int32)
    return np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)


# ── Weather lookup ─────────────────────────────────────────────────────────

def load_weather_lookup(region: str):
    """Returns shared-memory tensor (max_pid, W) and n_weather."""
    df   = pd.read_parquet(CLEAN_DIR / f"era5_{region}_pixel_features.parquet")
    drop = [c for c in ERA5_META - {"pixel_id"} if c in df.columns]
    feat = df.drop(columns=drop)
    n_w  = feat.shape[1] - 1

    max_pid = int(feat["pixel_id"].max()) + 1
    lookup  = np.zeros((max_pid, n_w), dtype=np.float32)
    ids     = feat["pixel_id"].values.astype(int)
    vals    = feat.drop(columns=["pixel_id"]).values.astype(np.float32)
    lookup[ids] = vals

    lookup_t = torch.from_numpy(lookup).share_memory_()
    print(f"  Weather {region}: {len(feat)} pixels x {n_w} features")
    return lookup_t, n_w


# ── Sampling ───────────────────────────────────────────────────────────────

def sample_terrain(region: str, n_total: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_DIR / f"{region}_terrain_ready.parquet")
    df["clc_type"] = df["clc_type"].apply(lambda x: 0 if x in UNSEEN_CLC else x)

    anchor     = "risk_0_2m"
    n_cls      = df[anchor].nunique()
    majority_n = int(n_total * 0.4)
    minority_n = int(n_total * 0.6 / max(n_cls - 1, 1))

    rng   = np.random.default_rng(seed)
    parts = []
    for cls in sorted(df[anchor].unique()):
        grp = df[df[anchor] == cls]
        cap = majority_n if cls == 0 else minority_n
        idx = rng.choice(len(grp), size=min(len(grp), cap), replace=False)
        parts.append(grp.iloc[idx])

    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    dist   = sample[anchor].value_counts().sort_index().to_dict()
    print(f"  Sample {region}: {len(sample):,} rows | dist={dist}")
    return sample


# ── Dataset ────────────────────────────────────────────────────────────────

class TerrainPatchDataset(Dataset):
    """
    Yields (patch [C,P,P] float32, weather [W] float32, labels [5] int64).
    raster and weather_lookup are shared-memory tensors — safe for
    num_workers > 0 on Windows (no pickling of large arrays).
    """

    def __init__(self, raster_t, valid_t, row, col,
                 era5_idx, weather_t, targets, augment=False):
        self.raster  = raster_t   # (C, H, W) shared tensor
        self.valid   = valid_t    # (H, W) shared tensor
        self.row     = row        # (N,) int32 numpy
        self.col     = col        # (N,) int32 numpy
        self.e5idx   = era5_idx   # (N,) int32 numpy
        self.weather = weather_t  # (max_pid, W) shared tensor
        self.targets = targets    # (N, 5) int64 numpy
        self.augment = augment
        self.C = raster_t.shape[0]
        self.H = raster_t.shape[1]
        self.W = raster_t.shape[2]

    def __len__(self):
        return len(self.row)

    def __getitem__(self, i):
        r, c = int(self.row[i]), int(self.col[i])

        r0, r1 = max(0, r - HALF), min(self.H, r + HALF + 1)
        c0, c1 = max(0, c - HALF), min(self.W, c + HALF + 1)
        pr0 = HALF - (r - r0);  pr1 = pr0 + (r1 - r0)
        pc0 = HALF - (c - c0);  pc1 = pc0 + (c1 - c0)

        patch = torch.zeros(self.C, PATCH_SIZE, PATCH_SIZE)
        patch[:, pr0:pr1, pc0:pc1] = (
            self.raster[:, r0:r1, c0:c1]
            * self.valid[r0:r1, c0:c1].unsqueeze(0)
        )

        if self.augment:
            if torch.rand(1).item() > 0.5:
                patch = patch.flip(2)
            if torch.rand(1).item() > 0.5:
                patch = patch.flip(1)

        weather = self.weather[int(self.e5idx[i])]
        labels  = torch.from_numpy(self.targets[i])
        return patch, weather, labels


# ── Model ──────────────────────────────────────────────────────────────────

class FloodRiskCNN(nn.Module):
    """Terrain patch encoder + weather MLP → 5 independent risk depth heads."""

    def __init__(self, n_terrain_ch: int, n_weather: int,
                 n_classes: int = 5, n_targets: int = 5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_terrain_ch, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,           64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),                                                  # 15×15
            nn.Conv2d(64,           128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                                  # 7×7
            nn.Conv2d(128,          256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                          # 4×4 → 4096
        )

        self.weather_branch = nn.Sequential(
            nn.Linear(n_weather, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )

        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 64, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),               nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(256, n_classes) for _ in range(n_targets)])

    def forward(self, patch, weather):
        enc = self.encoder(patch).flatten(1)
        wf  = self.weather_branch(weather)
        x   = self.fc_shared(torch.cat([enc, wf], dim=1))
        return [h(x) for h in self.heads]


# ── Training helpers ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total = 0.0
    for patch, weather, labels in loader:
        patch   = patch.to(DEVICE, non_blocking=True)
        weather = weather.to(DEVICE, non_blocking=True)
        labels  = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=USE_AMP):          # fp16 forward pass
            logits = model(patch, weather)
            loss   = sum(criterion(logits[t], labels[:, t]) for t in range(N_TARGETS))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds = [[] for _ in range(N_TARGETS)]
    trues = [[] for _ in range(N_TARGETS)]
    for patch, weather, labels in loader:
        patch, weather = patch.to(DEVICE), weather.to(DEVICE)
        with autocast(enabled=USE_AMP):
            logits = model(patch, weather)
        for t in range(N_TARGETS):
            preds[t].append(logits[t].argmax(1).cpu().numpy())
            trues[t].append(labels[:, t].numpy())

    results = {}
    for t, tgt in enumerate(TARGETS):
        y_pred = np.concatenate(preds[t])
        y_true = np.concatenate(trues[t])
        k = cohen_kappa_score(y_true, y_pred)
        print(f"\n{tgt}  kappa={k:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        results[tgt] = round(k, 4)
    return results


def build_dataset(region, raster_data, w_tensor, n_samples, augment):
    raster_t, valid_t, y0, x0, dy, dx, H, W = raster_data
    sample = sample_terrain(region, n_samples)
    row, col = proj_to_rc(sample["proj_y"].values, sample["proj_x"].values,
                          y0, x0, dy, dx, H, W)
    era5_idx = sample["era5_pixel_idx"].values.astype(np.int32)
    targets  = sample[TARGETS].values.astype(np.int64)
    return TerrainPatchDataset(raster_t, valid_t, row, col, era5_idx,
                               w_tensor, targets, augment=augment)


# ── Main ───────────────────────────────────────────────────────────────────

def train():
    print("\nLoading rasters into shared memory...")
    sev_raster = load_raster("severn")
    nor_raster = load_raster("northumbria")

    print("\nLoading weather lookups into shared memory...")
    sev_w, n_weather = load_weather_lookup("severn")
    nor_w, _         = load_weather_lookup("northumbria")

    print("\nBuilding datasets...")
    train_ds = build_dataset("severn",      sev_raster, sev_w, TRAIN_SAMPLES, augment=True)
    test_ds  = build_dataset("northumbria", nor_raster, nor_w, TEST_SAMPLES,  augment=False)

    # Weighted sampler for class balance
    anchor_labels = train_ds.targets[:, 0]
    cls_counts    = np.bincount(anchor_labels, minlength=N_CLASSES).astype(float)
    cls_w         = 1.0 / (cls_counts + 1.0)
    samp_w        = torch.from_numpy(cls_w[anchor_labels]).float()
    sampler = WeightedRandomSampler(samp_w, num_samples=len(train_ds), replacement=True)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False,   **loader_kwargs)

    model     = FloodRiskCNN(N_TERRAIN_CH, n_weather).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(DEVICE), label_smoothing=0.05)
    scaler    = GradScaler(enabled=(DEVICE.type == "cuda"))  # AMP scaler

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}  |  batch={BATCH_SIZE}  workers={NUM_WORKERS}"
          f"  AMP={'on' if DEVICE.type == 'cuda' else 'off'}")

    best_kappa = -1.0
    history    = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{EPOCHS}  loss={loss:.4f}  lr={lr_now:.2e}")

        if epoch % 5 == 0 or epoch == EPOCHS:
            kappas = evaluate(model, test_loader)
            mean_k = float(np.mean(list(kappas.values())))
            history.append({"epoch": epoch, "loss": round(loss, 4),
                             **kappas, "mean_kappa": round(mean_k, 4)})
            if mean_k > best_kappa:
                best_kappa = mean_k
                torch.save(model.state_dict(), MODEL_DIR / "cnn_best.pt")
                print(f"  ★ New best mean kappa: {best_kappa:.4f}  (saved cnn_best.pt)")

    torch.save(model.state_dict(), MODEL_DIR / "cnn_final.pt")
    with open(RESULTS_DIR / "cnn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best mean kappa: {best_kappa:.4f}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})" if USE_AMP else "")
          + f"  AMP={'on' if USE_AMP else 'off'}")
    train()
