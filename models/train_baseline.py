"""
LightGBM baseline for flood risk prediction -- Group 8
Train on Severn, evaluate on Northumbria.
One model per risk depth (5 total).

Run from the repo root:
    python models/train_baseline.py
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR   = Path(__file__).parent
REPO_ROOT   = MODEL_DIR.parent
DATA_DIR    = REPO_ROOT / "Data"
CLEAN_DIR   = DATA_DIR / "cleaned"
RESULTS_DIR = MODEL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
TARGETS = ["risk_0_2m", "risk_0_3m", "risk_0_6m", "risk_0_9m", "risk_1_2m"]

NON_FEAT = {"proj_y", "proj_x", "era5_pixel_idx"} | set(TARGETS)
ERA5_META = {"pixel_id", "proj_y", "proj_x", "region"}

MINORITY_CAP = 500_000   # max pixels per non-zero class in training sample
MAJORITY_CAP = 800_000   # max pixels for class 0

UNSEEN_CLC = {332}       # clc_type unseen in Severn, appears only in Northumbria

LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


# ── Data loading ───────────────────────────────────────────────────────────

def load_terrain(region: str) -> pd.DataFrame:
    """Load terrain parquet only (14 cols, ~2 GB). Targets are here."""
    df = pd.read_parquet(CLEAN_DIR / f"{region}_terrain_ready.parquet")
    df["clc_type"] = df["clc_type"].apply(lambda x: 0 if x in UNSEEN_CLC else x)
    print(f"  terrain {region}: {len(df):,} rows x {df.shape[1]} cols")
    return df


def load_weather(region: str) -> pd.DataFrame:
    """Load ERA5 pixel feature lookup (212 or 119 rows x ~81 cols)."""
    df = pd.read_parquet(CLEAN_DIR / f"era5_{region}_pixel_features.parquet")
    drop = [c for c in ERA5_META - {"pixel_id"} if c in df.columns]
    return df.drop(columns=drop)


def merge_weather(terrain_sample: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join weather features onto a terrain sample via era5_pixel_idx."""
    merged = terrain_sample.merge(
        weather, left_on="era5_pixel_idx", right_on="pixel_id", how="left"
    )
    return merged.drop(columns=["pixel_id"])


# ── Sampling (operates on terrain only — much cheaper) ─────────────────────

def stratified_sample(terrain: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Sample from terrain (14 cols) before merging weather.
    Caps class 0 at MAJORITY_CAP, each non-zero class at MINORITY_CAP.
    """
    parts = []
    for cls in terrain[target].unique():
        grp = terrain[terrain[target] == cls]
        cap = MAJORITY_CAP if cls == 0 else MINORITY_CAP
        parts.append(grp.sample(min(len(grp), cap), random_state=42))
    sampled = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    dist = sampled[target].value_counts().sort_index().to_dict()
    print(f"  Sample: {len(sampled):,} rows | dist={dist}")
    return sampled


# ── Training ───────────────────────────────────────────────────────────────

def train_and_evaluate():
    print("\nLoading terrain (Severn — train)...")
    train_terrain = load_terrain("severn")

    print("\nLoading terrain (Northumbria — test)...")
    test_terrain = load_terrain("northumbria")

    print("\nLoading weather features...")
    weather_sev = load_weather("severn")
    weather_nor = load_weather("northumbria")

    # Merge weather onto full test set (21M × 14 cols → 21M × ~83 cols, ~6 GB)
    # Do this once and reuse for all 5 models.
    print("\nMerging weather onto test set...")
    test_df = merge_weather(test_terrain, weather_nor)
    del test_terrain

    feat_cols = [c for c in test_df.columns if c not in NON_FEAT and c not in ERA5_META]
    print(f"Feature columns: {len(feat_cols)}")

    summary = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {target}")
        print(f"{'='*60}")

        # Sample terrain first (cheap: 14 cols), then merge weather onto sample
        sample_terrain = stratified_sample(train_terrain, target)
        sample = merge_weather(sample_terrain, weather_sev)
        del sample_terrain

        X_train = sample[feat_cols].values.astype(np.float32)
        y_train = sample[target].values
        y_test  = test_df[target].values
        del sample

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)
        del X_train, y_train

        # Predict in chunks to avoid allocating the full 13 GB float64 array
        CHUNK = 500_000
        chunks = []
        for start in range(0, len(test_df), CHUNK):
            X_chunk = test_df.iloc[start:start+CHUNK][feat_cols].values.astype(np.float32)
            chunks.append(model.predict(X_chunk))
        y_pred = np.concatenate(chunks)
        kappa  = cohen_kappa_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        print(f"  Cohen's Kappa: {kappa:.4f}")

        model_path = MODEL_DIR / f"lgbm_{target}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved: {model_path.name}")

        fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        fi.to_csv(RESULTS_DIR / f"feature_importance_{target}.csv")

        summary[target] = {
            "kappa":          round(kappa, 4),
            "top10_features": fi.head(10).index.tolist(),
        }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\nAll models trained. Summary:")
    for t, v in summary.items():
        print(f"  {t}: kappa={v['kappa']}  top feat={v['top10_features'][0]}")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    train_and_evaluate()
