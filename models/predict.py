"""
Load saved LightGBM models, run predictions on Northumbria,
save predictions parquet, and plot actual vs predicted risk maps.

Run from repo root:
    python models/predict.py
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent
REPO_ROOT = MODEL_DIR.parent
DATA_DIR  = REPO_ROOT / "Data"
CLEAN_DIR = DATA_DIR / "cleaned"
RESULTS_DIR = MODEL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGETS      = ["risk_0_2m", "risk_0_3m", "risk_0_6m", "risk_0_9m", "risk_1_2m"]
ERA5_META    = {"pixel_id", "proj_y", "proj_x", "region"}
NON_FEAT     = {"proj_y", "proj_x", "era5_pixel_idx"} | set(TARGETS)
UNSEEN_CLC   = {332}


def load_region(region: str) -> pd.DataFrame:
    terrain = pd.read_parquet(CLEAN_DIR / f"{region}_terrain_ready.parquet")
    weather = pd.read_parquet(CLEAN_DIR / f"era5_{region}_pixel_features.parquet")
    weather_lookup = weather.drop(columns=[c for c in ERA5_META - {"pixel_id"} if c in weather.columns])
    df = terrain.merge(weather_lookup, left_on="era5_pixel_idx", right_on="pixel_id", how="left")
    df = df.drop(columns=["pixel_id"])
    df["clc_type"] = df["clc_type"].apply(lambda x: 0 if x in UNSEEN_CLC else x)
    return df


def predict_all(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    X = df[feat_cols].values
    preds = {}
    for target in TARGETS:
        model_path = MODEL_DIR / f"lgbm_{target}.pkl"
        if not model_path.exists():
            print(f"  Model not found: {model_path.name} — skipping")
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        preds[f"pred_{target}"] = model.predict(X)
        print(f"  Predicted {target}")

    coord_cols = ["proj_y", "proj_x"] + TARGETS
    return pd.concat(
        [df[coord_cols].reset_index(drop=True), pd.DataFrame(preds)],
        axis=1,
    )


def plot_risk_map(result_df: pd.DataFrame, target: str = "risk_0_2m", region: str = "northumbria"):
    pred_col = f"pred_{target}"
    if pred_col not in result_df.columns:
        print(f"  No predictions for {target} — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, col, title in zip(axes, [target, pred_col], ["Actual", "Predicted"]):
        sc = ax.scatter(
            result_df["proj_x"], result_df["proj_y"],
            c=result_df[col], cmap="YlOrRd",
            s=0.02, vmin=0, vmax=4, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Risk class (0=none … 4=high)")
        ax.set_title(f"{title}  |  {target}  |  {region.capitalize()}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlabel("proj_x (EPSG:3035)")
        ax.set_ylabel("proj_y (EPSG:3035)")

    plt.tight_layout()
    out = RESULTS_DIR / f"map_{target}_{region}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved map: {out.name}")


if __name__ == "__main__":
    region = "northumbria"
    print(f"Loading {region}...")
    df = load_region(region)

    feat_cols = [c for c in df.columns if c not in NON_FEAT and c not in ERA5_META]
    print(f"Feature columns: {len(feat_cols)}")

    print("\nRunning predictions...")
    result_df = predict_all(df, feat_cols)

    out_parquet = RESULTS_DIR / f"{region}_predictions.parquet"
    result_df.to_parquet(out_parquet, index=False)
    print(f"\nSaved predictions: {out_parquet.name}  shape={result_df.shape}")

    print("\nPlotting risk maps...")
    for t in TARGETS:
        plot_risk_map(result_df, target=t, region=region)

    print("\nDone.")
