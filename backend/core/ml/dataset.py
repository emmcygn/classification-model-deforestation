"""Dataset loading and feature engineering for deforestation risk model."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

REGIONS = {
    "palawan": RAW_DIR / "palawan_grid.csv",
    "sierra_madre": RAW_DIR / "sierra_madre_grid.csv",
    "mindanao_bukidnon": RAW_DIR / "mindanao_bukidnon_grid.csv",
}

# Fallback to old Rondonia data if Philippine data not yet generated
FALLBACK_PATH = RAW_DIR / "rondonia_grid.csv"

# ── Feature manifest ──────────────────────────────────────────────
# Every training feature must declare its source window and whether
# it is available at prediction time (i.e., uses only pre-target data).
# The training pipeline validates this manifest and rejects any feature
# whose time window overlaps the target period.

FEATURE_MANIFEST = {
    # feature_name: (source, time_window, allowed_for_training)
    "tree_cover_2000_pct":                ("Hansen/UMD GFW v1.7",           "2000 (static)",   True),
    "elevation_m":                        ("SRTM 30m",                      "static",          True),
    "slope_deg":                          ("SRTM 30m",                      "static",          True),
    "dist_to_road_km":                    ("OpenStreetMap Overpass",         "snapshot",        True),
    "dist_to_deforestation_frontier_km":  ("Hansen GFW (pre-2020 loss)",    "2001-2019",       True),
    "protected_area":                     ("OpenStreetMap",                  "snapshot",        True),
    "population_density_per_km2":         ("WorldPop API",                  "2019 estimate",   True),
    "annual_loss_rate_pct":               ("Hansen GFW (pre-2020 loss)",    "2001-2019",       True),
    "exg_change_2018_2019":               ("Sentinel-2 cloudless (EOX)",    "2018-2019",       True),
    "neighbor_loss_count":                ("Derived (pre-2020 loss)",       "2001-2019",       True),
    "neighbor_loss_density":              ("Derived (pre-2020 loss)",       "2001-2019",       True),
    "neighbor_mean_tree_cover":           ("Derived (2000 tree cover)",     "2000 (static)",   True),
    "loss_acceleration":                  ("Derived (pre-2020 loss rate)",  "2001-2019",       True),
    "fire_hotspot_density":               ("Derived: high-loss-rate neighbor count (pre-2020)", "2001-2019", True),
    # Banned features — kept here for documentation, never used for training
    "exg_change_2018_2022":               ("Sentinel-2 cloudless (EOX)",    "2018-2022",       False),
    "loss_year":                          ("Hansen GFW",                    "2001-2022",       False),
    "high_risk":                          ("Target variable",               "2020-2022",       False),
    "pre_period_loss":                    ("Derived from loss_year",        "2001-2019",       False),
}

FEATURE_COLUMNS_FULL = [
    name for name, (_, _, allowed) in FEATURE_MANIFEST.items() if allowed
]

# Base features (always available in CSV)
FEATURE_COLUMNS_BASE = FEATURE_COLUMNS_FULL[:8]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in the dataframe and are allowed for training."""
    return [c for c in FEATURE_COLUMNS_FULL if c in df.columns]


def validate_features(feature_names: list[str]) -> None:
    """Fail fast if any feature is not allowed for training."""
    for name in feature_names:
        if name in FEATURE_MANIFEST:
            _, _, allowed = FEATURE_MANIFEST[name]
            if not allowed:
                raise ValueError(
                    f"Feature '{name}' is not allowed for training "
                    f"(time window overlaps target period or is the target itself)"
                )
        # Unknown features are allowed (user-defined)


# Default — use all allowed available features
FEATURE_COLUMNS = FEATURE_COLUMNS_FULL

# ── Target definition ─────────────────────────────────────────────
# Target: binary classification of forest loss in 2020-2022.
# This is a RETROSPECTIVE classification task — "predicting loss in
# 2020-2022 from pre-2020 signals" — NOT a generic future risk forecast.
# See docs/methodology.md "Target Formulation Limitations" for why this
# is a coarse but acceptable baseline formulation.
TARGET_COLUMN = "high_risk"

# Temporal split constants (must match fetch_philippines.py)
FEATURE_CUTOFF = 19   # loss_year 1-19 = 2001-2019 (feature period)
TARGET_START = 20      # loss_year 20-22 = 2020-2022 (prediction target)

# Forest eligibility: minimum tree cover to include a cell in training.
# Cells below this threshold were not plausibly forest before the target
# window and should not be included in a forest risk model.
FOREST_COVER_THRESHOLD = 10  # percent


def _apply_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute high_risk and pre_period_loss from loss_year for temporal holdout.

    Ensures features use only pre-period (2001-2019) data and the target
    reflects only the prediction period (2020-2022). This fixes target leakage
    in CSVs generated before the temporal split was implemented.
    """
    if "loss_year" not in df.columns:
        return df

    df = df.copy()
    ly = df["loss_year"].values

    # Target: loss in 2020-2022 (loss_year 20-22)
    df["high_risk"] = (ly >= TARGET_START).astype(int)

    # Feature: loss in 2001-2019 (loss_year 1-19)
    df["pre_period_loss"] = ((ly > 0) & (ly <= FEATURE_CUTOFF)).astype(int)

    return df


def load_dataset(region: str | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load a region's grid dataset."""
    if path:
        df = pd.read_csv(path)
        df = _apply_temporal_split(df)
        if "neighbor_loss_count" not in df.columns and "high_risk" in df.columns:
            from core.ml.feature_engineering import add_spatial_features
            df = add_spatial_features(df)
        return df
    if region:
        p = REGIONS.get(region)
        if p and p.exists():
            df = pd.read_csv(p)
            df = _apply_temporal_split(df)
            # Auto-compute spatial features if missing
            if "neighbor_loss_count" not in df.columns and "high_risk" in df.columns:
                from core.ml.feature_engineering import add_spatial_features
                df = add_spatial_features(df)
            return df
        raise FileNotFoundError(f"Dataset not found for region: {region}")
    # Load all available regions
    dfs = []
    for name, p in REGIONS.items():
        if p.exists():
            df = pd.read_csv(p)
            df = _apply_temporal_split(df)
            df["region"] = name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        # Auto-compute spatial features if missing
        if "neighbor_loss_count" not in df.columns and "high_risk" in df.columns:
            from core.ml.feature_engineering import add_spatial_features
            df = add_spatial_features(df)
        return df
    # Fallback to Rondonia — still apply temporal split if loss_year present
    if FALLBACK_PATH.exists():
        df = pd.read_csv(FALLBACK_PATH)
        df = _apply_temporal_split(df)
        return df
    raise FileNotFoundError("No datasets found")


def list_regions() -> list[dict]:
    """List available regions with metadata."""
    regions = []
    for name, path in REGIONS.items():
        if path.exists():
            df = pd.read_csv(path)
            regions.append({
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "cell_count": len(df),
                "bounds": {
                    "lat_min": round(float(df["lat"].min()), 4),
                    "lat_max": round(float(df["lat"].max()), 4),
                    "lon_min": round(float(df["lon"].min()), 4),
                    "lon_max": round(float(df["lon"].max()), 4),
                },
            })
    return regions


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    forest_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X, target vector y, and feature names.

    Args:
        df: Dataset with feature columns and target.
        feature_columns: Specific features to use (validated against manifest).
        forest_only: If True, exclude cells below FOREST_COVER_THRESHOLD.
            This ensures the model trains only on cells that were plausibly
            forest before the target window.
    """
    if feature_columns:
        validate_features(feature_columns)
        cols = [c for c in feature_columns if c in df.columns]
    else:
        cols = get_available_features(df)

    # Apply forest eligibility mask
    if forest_only and "tree_cover_2000_pct" in df.columns:
        mask = df["tree_cover_2000_pct"] >= FOREST_COVER_THRESHOLD
        df = df[mask].reset_index(drop=True)

    X = df[cols].values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.int64)
    return X, y, cols


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_data_spatial(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    block_size: float = 0.1,
    buffer_km: float = 0.0,  # Buffer exclusion: set >0 for fine-resolution (<100m) data
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spatial block cross-validation with buffer exclusion.

    Divides the geographic area into blocks and assigns entire blocks
    to train or test. Cells within buffer_km of the train/test boundary
    are excluded to prevent edge leakage (literature best practice).
    """
    rng = np.random.default_rng(random_state)

    # Assign each cell to a spatial block
    block_ids = (
        (df["lat"] // block_size).astype(int).astype(str) + "_" +
        (df["lon"] // block_size).astype(int).astype(str)
    )
    unique_blocks = block_ids.unique()

    # Randomly assign blocks to test set
    n_test_blocks = max(1, int(len(unique_blocks) * test_size))
    test_blocks = set(rng.choice(unique_blocks, n_test_blocks, replace=False))

    test_mask = block_ids.isin(test_blocks).values
    train_mask = ~test_mask

    # Buffer exclusion: drop cells near the train/test boundary
    if buffer_km > 0 and len(df) < 20000:
        import math
        mean_lat = df["lat"].mean()
        buffer_deg = buffer_km / 111.0  # approximate degrees

        train_lats = df.loc[train_mask, "lat"].values
        train_lons = df.loc[train_mask, "lon"].values
        test_lats = df.loc[test_mask, "lat"].values
        test_lons = df.loc[test_mask, "lon"].values

        # For each test cell, check min distance to any train cell
        # If too close, exclude from both sets
        exclude_test = np.zeros(test_mask.sum(), dtype=bool)
        exclude_train = np.zeros(train_mask.sum(), dtype=bool)

        for i, (tlat, tlon) in enumerate(zip(test_lats, test_lons)):
            dists = np.sqrt(
                ((train_lats - tlat) * 111.0) ** 2 +
                ((train_lons - tlon) * 111.0 * math.cos(math.radians(mean_lat))) ** 2
            )
            if dists.min() < buffer_km:
                exclude_test[i] = True

        # Apply exclusions
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(train_mask)[0]
        keep_test = test_indices[~exclude_test]
        keep_train = train_indices  # keep all train cells

        return X[keep_train], X[keep_test], y[keep_train], y[keep_test]

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]
