"""Pipeline API routes — dataset info, training, evaluation, run history."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np

from core.ml.dataset import load_dataset, prepare_features, split_data, FEATURE_COLUMNS, get_available_features
from core.ml.training import train_model, save_model, load_model
from core.ml.evaluation import evaluate_model
from core.ml.registry import RunRegistry
from core.ml.annotations import AnnotationStore
import pandas as pd

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "registry.db"

registry = RunRegistry(DB_PATH)
annotation_store = AnnotationStore(DATA_DIR / "annotations.db")


class TrainRequest(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    test_size: float = 0.2
    feature_columns: list[str] | None = None
    region: str | None = None
    spatial_split: bool = True
    model_type: str = "random_forest"


@router.on_event("startup")
async def startup():
    await registry.init()
    await annotation_store.init()


@router.get("/dataset")
async def get_dataset_info(region: str | None = None):
    """Return dataset summary."""
    df = load_dataset(region=region)
    available_features = get_available_features(df)
    feature_stats = {}
    for col in available_features:
        series = df[col]
        feature_stats[col] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
        }

    return {
        "row_count": len(df),
        "feature_columns": available_features,
        "feature_stats": feature_stats,
        "class_distribution": {
            "low_risk": int((df["high_risk"] == 0).sum()),
            "high_risk": int((df["high_risk"] == 1).sum()),
        },
        "geo_bounds": {
            "lat_min": round(float(df["lat"].min()), 4),
            "lat_max": round(float(df["lat"].max()), 4),
            "lon_min": round(float(df["lon"].min()), 4),
            "lon_max": round(float(df["lon"].max()), 4),
        },
    }


@router.get("/dataset/sample")
async def get_dataset_sample(n: int = 100, region: str | None = None):
    """Return a sample of the dataset."""
    df = load_dataset(region=region)
    sample = df.sample(min(n, len(df)), random_state=42)
    return sample.to_dict(orient="records")


@router.post("/train")
async def train(req: TrainRequest):
    """Train a model with given hyperparameters.

    Defaults to spatial CV split for honest evaluation. Persists test
    indices and dataset hash for full reproducibility.
    """
    import hashlib
    import numpy as np

    df_raw = load_dataset(region=req.region)
    features = req.feature_columns or FEATURE_COLUMNS
    X, y, feature_names = prepare_features(df_raw, features, forest_only=True)

    # Dataset hash for reproducibility
    dataset_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:16]

    # Split with index tracking
    indices = np.arange(len(X))
    if req.spatial_split:
        from core.ml.dataset import split_data_spatial, FOREST_COVER_THRESHOLD
        # Rebuild the filtered df for spatial block assignment
        if "tree_cover_2000_pct" in df_raw.columns:
            df_filtered = df_raw[df_raw["tree_cover_2000_pct"] >= FOREST_COVER_THRESHOLD].reset_index(drop=True)
        else:
            df_filtered = df_raw
        X_train, X_test, y_train, y_test = split_data_spatial(df_filtered, X, y, test_size=req.test_size)
        # Recover test indices
        from sklearn.model_selection import train_test_split as _unused
        test_hashes = {tuple(row) for row in X_test}
        test_idx = [int(i) for i in range(len(X)) if tuple(X[i]) in test_hashes]
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            indices, test_size=req.test_size, random_state=42, stratify=y
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_idx = [int(i) for i in test_idx]

    model = train_model(
        X_train, y_train,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        min_samples_split=req.min_samples_split,
        min_samples_leaf=req.min_samples_leaf,
        model_type=req.model_type,
    )

    metrics = evaluate_model(model, X_test, y_test, feature_names)

    from core.ml.evaluation import find_optimal_threshold
    threshold_analysis = find_optimal_threshold(model, X_test, y_test)

    # Auto-compare with alternative model
    alt_type = "xgboost" if req.model_type == "random_forest" else "random_forest"
    try:
        alt_model = train_model(X_train, y_train, n_estimators=req.n_estimators,
                                max_depth=req.max_depth, model_type=alt_type)
        alt_metrics = evaluate_model(alt_model, X_test, y_test, feature_names)
        comparison = {
            "primary_model": req.model_type,
            "primary_f1": metrics["f1"],
            "alternative_model": alt_type,
            "alternative_f1": alt_metrics["f1"],
            "recommendation": req.model_type if metrics["f1"] >= alt_metrics["f1"] else alt_type,
        }
    except Exception:
        comparison = None

    params = req.model_dump()
    params["threshold_analysis"] = threshold_analysis
    params["model_comparison"] = comparison

    run_id = await registry.save_run(
        params=params,
        metrics=metrics,
        feature_names=feature_names,
        model_path="",
        test_indices=test_idx,
        dataset_hash=dataset_hash,
    )

    model_path = save_model(model, MODELS_DIR, run_id)

    import aiosqlite
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("UPDATE runs SET model_path = ? WHERE run_id = ?", (str(model_path), run_id))
        await db.commit()

    return {"run_id": run_id, "metrics": metrics, "threshold_analysis": threshold_analysis, "model_comparison": comparison}


@router.post("/retrain")
async def retrain_with_annotations(req: TrainRequest):
    """Retrain model using human annotations to override ground-truth labels.

    Annotations marked 'reject' on high-risk cells flip their label to 0 (false alarm).
    Annotations marked 'accept' on low-risk cells flip their label to 1 (confirmed risk).
    This creates a human-curated training set for improved model accuracy.
    """
    df = load_dataset(region=req.region)
    features = req.feature_columns or FEATURE_COLUMNS

    # Load annotations and filter to this region's cells
    annotations = await annotation_store.list_annotations()
    if req.region:
        region_df = load_dataset(region=req.region)
        region_lats = set(round(lat, 3) for lat in region_df["lat"])
        region_lons = set(round(lon, 3) for lon in region_df["lon"])
        annotations = [a for a in annotations if round(a["lat"], 3) in region_lats and round(a["lon"], 3) in region_lons]

    overrides = 0
    for ann in annotations:
        # Find the closest cell in the dataset
        mask = ((df["lat"] - ann["lat"]).abs() < 0.001) & ((df["lon"] - ann["lon"]).abs() < 0.001)
        if mask.any():
            idx = mask.idxmax()
            if ann["verdict"] == "reject":
                # Official says this is NOT a real threat — set label to 0
                df.loc[idx, "high_risk"] = 0
                overrides += 1
            elif ann["verdict"] == "accept" and df.loc[idx, "high_risk"] == 0:
                # Official confirms this IS a threat even though model/data said no — set label to 1
                df.loc[idx, "high_risk"] = 1
                overrides += 1

    X, y, feature_names = prepare_features(df, features)

    if req.spatial_split:
        from core.ml.dataset import split_data_spatial
        X_train, X_test, y_train, y_test = split_data_spatial(df, X, y, test_size=req.test_size)
    else:
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=req.test_size)

    model = train_model(
        X_train, y_train,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        min_samples_split=req.min_samples_split,
        min_samples_leaf=req.min_samples_leaf,
    )

    metrics = evaluate_model(model, X_test, y_test, feature_names)

    params = req.model_dump()
    params["annotation_overrides"] = overrides

    run_id = await registry.save_run(
        params=params,
        metrics=metrics,
        feature_names=feature_names,
        model_path="",
    )

    model_path = save_model(model, MODELS_DIR, run_id)

    import aiosqlite
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("UPDATE runs SET model_path = ? WHERE run_id = ?", (str(model_path), run_id))
        await db.commit()

    # Also train a baseline model WITHOUT annotation overrides for comparison
    df_baseline = load_dataset(region=req.region)
    X_base, y_base, _ = prepare_features(df_baseline, features)
    # Use same split method as retrained model for fair comparison
    if req.spatial_split:
        from core.ml.dataset import split_data_spatial
        X_train_base, X_test_base, y_train_base, y_test_base = split_data_spatial(df_baseline, X_base, y_base, test_size=req.test_size)
    else:
        X_train_base, X_test_base, y_train_base, y_test_base = split_data(X_base, y_base, test_size=req.test_size)
    model_baseline = train_model(X_train_base, y_train_base, n_estimators=req.n_estimators, max_depth=req.max_depth)
    metrics_baseline = evaluate_model(model_baseline, X_test_base, y_test_base, features)

    return {
        "run_id": run_id,
        "metrics": metrics,
        "metrics_baseline": metrics_baseline,
        "annotation_overrides": overrides,
        "total_annotations": len(annotations),
        "improvement": {
            "accuracy_delta": round(metrics["accuracy"] - metrics_baseline["accuracy"], 4),
            "f1_delta": round(metrics["f1"] - metrics_baseline["f1"], 4),
            "precision_delta": round(metrics["precision"] - metrics_baseline["precision"], 4),
            "recall_delta": round(metrics["recall"] - metrics_baseline["recall"], 4),
        },
    }


@router.get("/runs")
async def list_runs():
    """List all training runs."""
    return await registry.list_runs()


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get a specific training run."""
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
