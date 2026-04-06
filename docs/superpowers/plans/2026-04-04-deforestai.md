# DeforestAI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deforestation risk intelligence platform with an interactive map explorer and a real ML training pipeline UI.

**Architecture:** FastAPI backend with two route groups (explorer, pipeline). React+Vite+TypeScript frontend with two pages (Explorer, Pipeline). scikit-learn Random Forest for risk prediction with SHAP explainability. SQLite for run history/model registry. Claude API for narrative report synthesis.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, scikit-learn, SHAP, SQLite, anthropic SDK, React 18, Vite, TypeScript, Leaflet, Recharts, TailwindCSS

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/main.py`
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
venv/
.venv/
node_modules/
dist/
.env
*.db
backend/data/models/*.joblib
backend/data/runs/*.json
.superpowers/
```

- [ ] **Step 2: Create backend requirements.txt**

```txt
fastapi==0.115.0
uvicorn==0.30.6
scikit-learn==1.5.2
shap==0.46.0
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
anthropic==0.34.0
pydantic==2.9.2
aiosqlite==0.20.0
geopy==2.4.1
```

- [ ] **Step 3: Create backend/main.py with hello-world endpoint**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DeforestAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Create Python venv and install dependencies**

Run:
```bash
cd backend && python -m venv venv && venv/Scripts/activate && pip install -r requirements.txt
```
(On Windows use `venv\Scripts\activate`, on Unix use `source venv/bin/activate`)

- [ ] **Step 5: Test backend starts**

Run:
```bash
cd backend && venv/Scripts/python -m uvicorn main:app --port 8000 &
curl http://localhost:8000/api/health
```
Expected: `{"status":"ok"}`

- [ ] **Step 6: Scaffold frontend with Vite + React + TypeScript**

Run:
```bash
cd frontend && npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install react-router-dom leaflet react-leaflet recharts @types/leaflet
```

- [ ] **Step 7: Configure Tailwind with Vite plugin**

Update `frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

Replace `frontend/src/index.css` with:
```css
@import "tailwindcss";
```

- [ ] **Step 8: Create App.tsx with router shell**

```tsx
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";

function Explorer() {
  return <div className="p-8"><h1 className="text-2xl font-bold">Explorer</h1></div>;
}

function Pipeline() {
  return <div className="p-8"><h1 className="text-2xl font-bold">Pipeline</h1></div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-6">
          <span className="text-lg font-bold text-emerald-400">DeforestAI</span>
          <NavLink to="/" className={({ isActive }) => isActive ? "text-emerald-400" : "text-gray-400 hover:text-gray-200"}>Explorer</NavLink>
          <NavLink to="/pipeline" className={({ isActive }) => isActive ? "text-emerald-400" : "text-gray-400 hover:text-gray-200"}>Pipeline</NavLink>
        </nav>
        <Routes>
          <Route path="/" element={<Explorer />} />
          <Route path="/pipeline" element={<Pipeline />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
```

- [ ] **Step 9: Test frontend starts**

Run:
```bash
cd frontend && npm run dev
```
Expected: Opens on http://localhost:5173, shows nav with Explorer/Pipeline links

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: scaffold project with FastAPI backend and React frontend"
```

---

### Task 2: Sample Dataset Generation

**Files:**
- Create: `backend/core/ml/generate_dataset.py`
- Create: `backend/data/raw/` (directory)

- [ ] **Step 1: Create the dataset generator**

```python
"""Generate a synthetic but realistic deforestation risk dataset.

Covers a grid over Rondônia, Brazil (~10.5°S to ~13.5°S, ~60°W to ~63.5°W).
Each row is a 0.05° grid cell (~5.5km) with features derived from
known distributions of Hansen/GFW statistics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Rondônia bounding box
LAT_MIN, LAT_MAX = -13.5, -10.5
LON_MIN, LON_MAX = -63.5, -60.0
STEP = 0.05  # ~5.5km grid cells


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    lats = np.arange(LAT_MIN, LAT_MAX, STEP)
    lons = np.arange(LON_MIN, LON_MAX, STEP)
    grid = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)

    n = len(grid)

    # --- Features ---
    tree_cover_2000 = rng.beta(5, 2, n) * 100  # mostly forested
    elevation_m = rng.normal(200, 80, n).clip(50, 600)
    slope_deg = rng.exponential(3, n).clip(0, 30)
    dist_to_road_km = rng.exponential(15, n).clip(0.1, 200)
    dist_to_deforestation_frontier_km = rng.exponential(20, n).clip(0.1, 300)
    protected_area = rng.choice([0, 1], n, p=[0.7, 0.3])
    population_density = rng.exponential(10, n).clip(0, 200)

    # Historical annual loss rate (%/yr) — higher near roads, lower in protected
    base_loss = rng.exponential(1.5, n)
    road_effect = np.exp(-dist_to_road_km / 10) * 3
    protection_effect = protected_area * (-1.5)
    annual_loss_rate = (base_loss + road_effect + protection_effect).clip(0, 15)

    # --- Target: did significant loss occur in most recent 3 years? ---
    # Logistic probability based on features
    logit = (
        -2.0
        + 0.3 * annual_loss_rate
        - 0.01 * dist_to_road_km
        + 0.005 * population_density
        - 0.003 * elevation_m
        - 0.8 * protected_area
        - 0.01 * dist_to_deforestation_frontier_km
        + 0.02 * slope_deg
    )
    prob = 1 / (1 + np.exp(-logit))
    high_risk = rng.binomial(1, prob)

    df = pd.DataFrame({
        "lat": grid[:, 0],
        "lon": grid[:, 1],
        "tree_cover_2000_pct": np.round(tree_cover_2000, 1),
        "elevation_m": np.round(elevation_m, 1),
        "slope_deg": np.round(slope_deg, 1),
        "dist_to_road_km": np.round(dist_to_road_km, 2),
        "dist_to_deforestation_frontier_km": np.round(dist_to_deforestation_frontier_km, 2),
        "protected_area": protected_area,
        "population_density_per_km2": np.round(population_density, 1),
        "annual_loss_rate_pct": np.round(annual_loss_rate, 2),
        "high_risk": high_risk,
    })

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    out_path = OUTPUT_DIR / "rondonia_grid.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} grid cells -> {out_path}")
    print(f"High risk: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run:
```bash
cd backend && venv/Scripts/python -m core.ml.generate_dataset
```
Expected: Prints row count and high_risk distribution. Creates `backend/data/raw/rondonia_grid.csv`.

- [ ] **Step 3: Create `__init__.py` files for packages**

Create empty files:
- `backend/core/__init__.py`
- `backend/core/ml/__init__.py`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add synthetic deforestation dataset generator for Rondônia"
```

---

### Task 3: ML Dataset Loading & Feature Engineering

**Files:**
- Create: `backend/core/ml/dataset.py`
- Create: `backend/tests/test_dataset.py`

- [ ] **Step 1: Write test for dataset loading**

```python
import pytest
from core.ml.dataset import load_dataset, prepare_features

def test_load_dataset():
    df = load_dataset()
    assert len(df) > 0
    assert "lat" in df.columns
    assert "high_risk" in df.columns

def test_prepare_features():
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(feature_names)
    assert len(y) == len(df)
    assert "lat" not in feature_names
    assert "lon" not in feature_names
    assert "high_risk" not in feature_names
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_dataset.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Implement dataset.py**

```python
"""Dataset loading and feature engineering for deforestation risk model."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "rondonia_grid.csv"

FEATURE_COLUMNS = [
    "tree_cover_2000_pct",
    "elevation_m",
    "slope_deg",
    "dist_to_road_km",
    "dist_to_deforestation_frontier_km",
    "protected_area",
    "population_density_per_km2",
    "annual_loss_rate_pct",
]

TARGET_COLUMN = "high_risk"


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw grid dataset."""
    p = path or RAW_PATH
    return pd.read_csv(p)


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X, target vector y, and feature names."""
    cols = feature_columns or FEATURE_COLUMNS
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
```

- [ ] **Step 4: Create `backend/tests/__init__.py`**

Empty file.

- [ ] **Step 5: Run tests**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_dataset.py -v
```
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add dataset loading and feature engineering module"
```

---

### Task 4: ML Training & Evaluation Core

**Files:**
- Create: `backend/core/ml/training.py`
- Create: `backend/core/ml/evaluation.py`
- Create: `backend/tests/test_training.py`

- [ ] **Step 1: Write training tests**

```python
import pytest
import numpy as np
from core.ml.dataset import load_dataset, prepare_features, split_data
from core.ml.training import train_model, save_model, load_model
from core.ml.evaluation import evaluate_model

@pytest.fixture
def data():
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, feature_names

def test_train_model(data):
    X_train, X_test, y_train, y_test, feature_names = data
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})

def test_evaluate_model(data):
    X_train, X_test, y_train, y_test, feature_names = data
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "confusion_matrix" in metrics
    assert "feature_importance" in metrics
    assert len(metrics["feature_importance"]) == len(feature_names)

def test_save_load_model(data, tmp_path):
    X_train, _, y_train, _, _ = data
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
    path = save_model(model, tmp_path, "test_model")
    loaded = load_model(path)
    assert np.array_equal(model.predict(X_train[:5]), loaded.predict(X_train[:5]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_training.py -v
```
Expected: FAIL — modules not found

- [ ] **Step 3: Implement training.py**

```python
"""Model training and persistence."""

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: RandomForestClassifier, directory: Path, name: str) -> Path:
    """Save model to disk. Returns the path to the saved file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> RandomForestClassifier:
    """Load a saved model from disk."""
    return joblib.load(path)
```

- [ ] **Step 4: Implement evaluation.py**

```python
"""Model evaluation and metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    importances = model.feature_importances_
    feature_importance = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    ]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
    }
```

- [ ] **Step 5: Run tests**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_training.py -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add ML training and evaluation modules"
```

---

### Task 5: SHAP Explainability

**Files:**
- Create: `backend/core/ml/explainability.py`
- Create: `backend/tests/test_explainability.py`

- [ ] **Step 1: Write explainability test**

```python
import pytest
from core.ml.dataset import load_dataset, prepare_features, split_data
from core.ml.training import train_model
from core.ml.explainability import explain_prediction, explain_summary_text

@pytest.fixture
def trained():
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, n_estimators=20, max_depth=5)
    return model, X_test, feature_names

def test_explain_prediction(trained):
    model, X_test, feature_names = trained
    result = explain_prediction(model, X_test[0:1], feature_names)
    assert "shap_values" in result
    assert "base_value" in result
    assert "prediction" in result
    assert len(result["shap_values"]) == len(feature_names)

def test_explain_summary_text(trained):
    model, X_test, feature_names = trained
    explanation = explain_prediction(model, X_test[0:1], feature_names)
    text = explain_summary_text(explanation, X_test[0], feature_names)
    assert isinstance(text, str)
    assert len(text) > 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_explainability.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement explainability.py**

```python
"""SHAP-based model explainability for individual predictions."""

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier


def explain_prediction(
    model: RandomForestClassifier,
    X_single: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compute SHAP values for a single prediction.

    Args:
        model: Trained RandomForest
        X_single: Shape (1, n_features)
        feature_names: List of feature names

    Returns:
        Dict with shap_values, base_value, prediction, and per-feature contributions.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)

    prediction = int(model.predict(X_single)[0])
    # For binary classification, shap_values is [array_class0, array_class1]
    # Use the values for the predicted class
    if isinstance(shap_values, list):
        sv = shap_values[prediction][0]
        base = float(explainer.expected_value[prediction])
    else:
        sv = shap_values[0]
        base = float(explainer.expected_value)

    contributions = [
        {
            "feature": name,
            "value": round(float(X_single[0, i]), 4),
            "shap_value": round(float(sv[i]), 4),
        }
        for i, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda x: -abs(x["shap_value"]))

    return {
        "prediction": prediction,
        "prediction_label": "High Risk" if prediction == 1 else "Low Risk",
        "base_value": round(base, 4),
        "shap_values": contributions,
    }


def explain_summary_text(
    explanation: dict,
    x_raw: np.ndarray,
    feature_names: list[str],
) -> str:
    """Generate a deterministic, templated explainability summary.

    Example output:
    "High Risk — driven by: road proximity (0.3km, +0.28),
     recent loss rate (4.2%/yr, +0.22), low elevation (120m, +0.15)"
    """
    label = explanation["prediction_label"]
    contribs = explanation["shap_values"]

    # Top 3 contributors by absolute SHAP value
    top = contribs[:3]

    parts = []
    for c in top:
        sign = "+" if c["shap_value"] >= 0 else ""
        parts.append(f"{c['feature']} ({c['value']}, {sign}{c['shap_value']})")

    drivers = ", ".join(parts)
    return f"{label} — driven by: {drivers}"
```

- [ ] **Step 4: Run tests**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_explainability.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add SHAP explainability for per-prediction explanations"
```

---

### Task 6: Model Registry & Run History (SQLite)

**Files:**
- Create: `backend/core/ml/registry.py`
- Create: `backend/tests/test_registry.py`

- [ ] **Step 1: Write registry tests**

```python
import pytest
import asyncio
from core.ml.registry import RunRegistry

@pytest.fixture
def registry(tmp_path):
    reg = RunRegistry(tmp_path / "test.db")
    asyncio.get_event_loop().run_until_complete(reg.init())
    return reg

def test_save_and_list_runs(registry):
    loop = asyncio.get_event_loop()
    run_id = loop.run_until_complete(registry.save_run(
        params={"n_estimators": 100, "max_depth": 10},
        metrics={"accuracy": 0.85, "f1": 0.82},
        feature_names=["f1", "f2"],
        model_path="/tmp/model.joblib",
    ))
    assert run_id is not None

    runs = loop.run_until_complete(registry.list_runs())
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["metrics"]["accuracy"] == 0.85

def test_get_run(registry):
    loop = asyncio.get_event_loop()
    run_id = loop.run_until_complete(registry.save_run(
        params={"n_estimators": 50},
        metrics={"accuracy": 0.90},
        feature_names=["f1"],
        model_path="/tmp/m.joblib",
    ))
    run = loop.run_until_complete(registry.get_run(run_id))
    assert run is not None
    assert run["params"]["n_estimators"] == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_registry.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement registry.py**

```python
"""Model registry and run history backed by SQLite."""

import json
import uuid
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path


class RunRegistry:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)

    async def init(self):
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    params TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    feature_names TEXT NOT NULL,
                    model_path TEXT NOT NULL
                )
            """)
            await db.commit()

    async def save_run(
        self,
        params: dict,
        metrics: dict,
        feature_names: list[str],
        model_path: str,
    ) -> str:
        """Save a training run. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO runs (run_id, created_at, params, metrics, feature_names, model_path) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, created_at, json.dumps(params), json.dumps(metrics), json.dumps(feature_names), model_path),
            )
            await db.commit()
        return run_id

    async def list_runs(self) -> list[dict]:
        """List all runs, newest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    async def get_run(self, run_id: str) -> dict | None:
        """Get a single run by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = await cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row) -> dict:
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "params": json.loads(row["params"]),
            "metrics": json.loads(row["metrics"]),
            "feature_names": json.loads(row["feature_names"]),
            "model_path": row["model_path"],
        }
```

- [ ] **Step 4: Run tests**

Run:
```bash
cd backend && venv/Scripts/python -m pytest tests/test_registry.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add SQLite-backed model registry and run history"
```

---

### Task 7: Pipeline API Routes

**Files:**
- Create: `backend/api/__init__.py`
- Create: `backend/api/routes/__init__.py`
- Create: `backend/api/routes/pipeline.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Implement pipeline routes**

```python
"""Pipeline API routes — dataset info, training, evaluation, run history."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np

from core.ml.dataset import load_dataset, prepare_features, split_data, FEATURE_COLUMNS
from core.ml.training import train_model, save_model, load_model
from core.ml.evaluation import evaluate_model
from core.ml.registry import RunRegistry

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "registry.db"

registry = RunRegistry(DB_PATH)


class TrainRequest(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    test_size: float = 0.2
    feature_columns: list[str] | None = None


@router.on_event("startup")
async def startup():
    await registry.init()


@router.get("/dataset")
async def get_dataset_info():
    """Return dataset summary: row count, feature stats, class distribution."""
    df = load_dataset()
    feature_stats = {}
    for col in FEATURE_COLUMNS:
        series = df[col]
        feature_stats[col] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
        }

    return {
        "row_count": len(df),
        "feature_columns": FEATURE_COLUMNS,
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
async def get_dataset_sample(n: int = 100):
    """Return a sample of the dataset for display."""
    df = load_dataset()
    sample = df.sample(min(n, len(df)), random_state=42)
    return sample.to_dict(orient="records")


@router.post("/train")
async def train(req: TrainRequest):
    """Train a model with given hyperparameters."""
    df = load_dataset()
    features = req.feature_columns or FEATURE_COLUMNS
    X, y, feature_names = prepare_features(df, features)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=req.test_size)

    model = train_model(
        X_train, y_train,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        min_samples_split=req.min_samples_split,
        min_samples_leaf=req.min_samples_leaf,
    )

    metrics = evaluate_model(model, X_test, y_test, feature_names)

    run_id = await registry.save_run(
        params=req.model_dump(),
        metrics=metrics,
        feature_names=feature_names,
        model_path="",  # placeholder, updated after save
    )

    model_path = save_model(model, MODELS_DIR, run_id)

    # Update with real path
    async with __import__("aiosqlite").connect(str(DB_PATH)) as db:
        await db.execute("UPDATE runs SET model_path = ? WHERE run_id = ?", (str(model_path), run_id))
        await db.commit()

    return {"run_id": run_id, "metrics": metrics}


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
```

- [ ] **Step 2: Register router in main.py**

Replace `backend/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.pipeline import router as pipeline_router

app = FastAPI(title="DeforestAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.include_router(pipeline_router)
```

- [ ] **Step 3: Create `__init__.py` files**

Create empty `backend/api/__init__.py` and `backend/api/routes/__init__.py`.

- [ ] **Step 4: Test the endpoint manually**

Run:
```bash
cd backend && venv/Scripts/python -m uvicorn main:app --reload --port 8000
```
Then in another terminal:
```bash
curl http://localhost:8000/api/pipeline/dataset
curl -X POST http://localhost:8000/api/pipeline/train -H "Content-Type: application/json" -d '{"n_estimators": 20, "max_depth": 5}'
curl http://localhost:8000/api/pipeline/runs
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add pipeline API routes (dataset, train, runs)"
```

---

### Task 8: Explorer API Routes

**Files:**
- Create: `backend/api/routes/explorer.py`
- Create: `backend/core/ai/__init__.py`
- Create: `backend/core/ai/analysis.py`
- Create: `backend/core/geo/__init__.py`
- Create: `backend/core/geo/lookup.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Implement geo lookup**

```python
"""Geocoding and region resolution."""

from geopy.geocoders import Nominatim


_geolocator = Nominatim(user_agent="deforestai-prototype")


def geocode(query: str) -> dict | None:
    """Geocode a search query to lat/lon/bounds."""
    location = _geolocator.geocode(query, exactly_one=True, viewbox=None)
    if not location:
        return None
    return {
        "lat": location.latitude,
        "lon": location.longitude,
        "display_name": location.address,
        "raw": location.raw,
    }
```

- [ ] **Step 2: Implement AI analysis (Claude report generation)**

```python
"""AI-powered report generation using Claude API."""

import os
import json
from anthropic import Anthropic

REPORT_SCHEMA = {
    "region_summary": "Area name, total grid cells, geographic bounds",
    "risk_distribution": "Count and % of high/medium/low risk cells",
    "top_risk_factors": "Aggregated feature importances across the region",
    "notable_data_points": "Auto-flagged anomalies and clusters",
    "trend_analysis": "Historical loss trajectory description",
    "narrative_synthesis": "Claude-generated readable prose tying it all together",
}


def generate_report(region_data: dict) -> dict:
    """Generate a structured analysis report.

    region_data should contain:
    - region_name: str
    - stats: dict of aggregated statistics
    - risk_distribution: dict with counts
    - top_features: list of feature importance dicts
    - notable_points: list of flagged anomalies
    """
    # Build deterministic sections first
    report = {
        "region_summary": {
            "name": region_data.get("region_name", "Selected Region"),
            "total_cells": region_data["stats"]["total_cells"],
            "bounds": region_data["stats"]["bounds"],
        },
        "risk_distribution": region_data["risk_distribution"],
        "top_risk_factors": region_data["top_features"],
        "notable_data_points": region_data["notable_points"],
    }

    # Use Claude for narrative synthesis only
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        report["narrative_synthesis"] = (
            "AI narrative unavailable — set ANTHROPIC_API_KEY environment variable. "
            "All other report sections are generated deterministically from model outputs."
        )
        return report

    client = Anthropic(api_key=api_key)
    prompt = f"""You are a deforestation risk analyst. Given the following structured data about a region, 
write a concise 2-3 paragraph narrative synthesis. Be specific, cite numbers from the data, 
and highlight the most actionable findings. Do not add information not present in the data.

Region: {report['region_summary']['name']}
Total cells analyzed: {report['region_summary']['total_cells']}
Risk distribution: {json.dumps(report['risk_distribution'])}
Top risk factors: {json.dumps(report['top_risk_factors'][:5])}
Notable findings: {json.dumps(report['notable_data_points'][:5])}

Write the narrative synthesis now:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    report["narrative_synthesis"] = message.content[0].text

    return report
```

- [ ] **Step 3: Implement explorer routes**

```python
"""Explorer API routes — map data, cell details, reports."""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import numpy as np

from core.ml.dataset import load_dataset, prepare_features, FEATURE_COLUMNS
from core.ml.training import load_model
from core.ml.explainability import explain_prediction, explain_summary_text
from core.ml.registry import RunRegistry
from core.geo.lookup import geocode
from core.ai.analysis import generate_report

router = APIRouter(prefix="/api/explorer", tags=["explorer"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "registry.db"
registry = RunRegistry(DB_PATH)


@router.get("/geocode")
async def geocode_search(q: str = Query(..., description="Search query")):
    """Geocode a location string."""
    result = geocode(q)
    if not result:
        raise HTTPException(status_code=404, detail="Location not found")
    return result


@router.get("/grid")
async def get_grid(run_id: str = Query(..., description="Model run ID")):
    """Get all grid cells with risk predictions from a specific model run."""
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    df = load_dataset()
    feature_names = run["feature_names"]
    X, _, _ = prepare_features(df, feature_names)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    cells = []
    for i in range(len(df)):
        cells.append({
            "lat": float(df.iloc[i]["lat"]),
            "lon": float(df.iloc[i]["lon"]),
            "prediction": int(predictions[i]),
            "risk_probability": round(float(probabilities[i]), 4),
        })

    return {"cells": cells, "run_id": run_id}


@router.get("/cell")
async def get_cell_detail(
    lat: float,
    lon: float,
    run_id: str,
):
    """Get detailed info for a specific grid cell including SHAP explanation."""
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model = load_model(Path(run["model_path"]))
    df = load_dataset()
    feature_names = run["feature_names"]

    # Find closest cell
    distances = ((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2).values
    idx = int(np.argmin(distances))

    X, _, _ = prepare_features(df, feature_names)
    x_single = X[idx:idx+1]

    explanation = explain_prediction(model, x_single, feature_names)
    summary_text = explain_summary_text(explanation, X[idx], feature_names)

    row = df.iloc[idx]
    features = {col: round(float(row[col]), 4) for col in feature_names}

    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "features": features,
        "explanation": explanation,
        "summary_text": summary_text,
    }


@router.post("/report")
async def generate_region_report(
    run_id: str,
    lat_min: float = Query(-13.5),
    lat_max: float = Query(-10.5),
    lon_min: float = Query(-63.5),
    lon_max: float = Query(-60.0),
):
    """Generate a structured analysis report for a region."""
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model = load_model(Path(run["model_path"]))
    df = load_dataset()
    feature_names = run["feature_names"]

    # Filter to region
    mask = (
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
    )
    region_df = df[mask]

    if len(region_df) == 0:
        raise HTTPException(status_code=400, detail="No data in selected region")

    X, _, _ = prepare_features(region_df, feature_names)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    high_count = int((predictions == 1).sum())
    low_count = int((predictions == 0).sum())
    total = len(predictions)

    # Aggregate feature importances
    importances = model.feature_importances_
    top_features = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    ]

    # Find notable data points
    notable = []
    high_risk_cells = np.where(predictions == 1)[0]
    if len(high_risk_cells) > 0:
        max_prob_idx = high_risk_cells[np.argmax(probabilities[high_risk_cells])]
        row = region_df.iloc[max_prob_idx]
        notable.append(
            f"Highest risk cell at ({row['lat']:.2f}, {row['lon']:.2f}) "
            f"with {probabilities[max_prob_idx]:.0%} probability"
        )
    if high_count > total * 0.3:
        notable.append(f"Region has elevated risk: {high_count/total:.0%} of cells are high-risk")

    region_data = {
        "region_name": f"Region ({lat_min:.1f} to {lat_max:.1f}N, {lon_min:.1f} to {lon_max:.1f}E)",
        "stats": {
            "total_cells": total,
            "bounds": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        },
        "risk_distribution": {
            "high_risk": high_count,
            "low_risk": low_count,
            "high_risk_pct": round(high_count / total * 100, 1),
        },
        "top_features": top_features,
        "notable_points": notable,
    }

    report = generate_report(region_data)
    return report
```

- [ ] **Step 4: Register explorer router in main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.pipeline import router as pipeline_router
from api.routes.explorer import router as explorer_router

app = FastAPI(title="DeforestAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.include_router(pipeline_router)
app.include_router(explorer_router)
```

- [ ] **Step 5: Create empty `__init__.py` files**

- `backend/core/ai/__init__.py`
- `backend/core/geo/__init__.py`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add explorer API routes (geocode, grid, cell detail, report)"
```

---

### Task 9: Frontend — Pipeline Dataset Tab

**Files:**
- Create: `frontend/src/pages/Pipeline.tsx`
- Create: `frontend/src/components/pipeline/DatasetTab.tsx`
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: Create API client**

```typescript
// frontend/src/lib/api.ts

const BASE = "/api";

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// Pipeline
export const getDatasetInfo = () => fetchJSON<DatasetInfo>("/pipeline/dataset");
export const getDatasetSample = (n = 100) => fetchJSON<Record<string, number>[]>(`/pipeline/dataset/sample?n=${n}`);
export const trainModel = (params: TrainParams) =>
  fetchJSON<TrainResult>("/pipeline/train", { method: "POST", body: JSON.stringify(params) });
export const listRuns = () => fetchJSON<Run[]>("/pipeline/runs");
export const getRun = (id: string) => fetchJSON<Run>(`/pipeline/runs/${id}`);

// Explorer
export const geocodeSearch = (q: string) => fetchJSON<GeocodeResult>(`/explorer/geocode?q=${encodeURIComponent(q)}`);
export const getGrid = (runId: string) => fetchJSON<GridResponse>(`/explorer/grid?run_id=${runId}`);
export const getCellDetail = (lat: number, lon: number, runId: string) =>
  fetchJSON<CellDetail>(`/explorer/cell?lat=${lat}&lon=${lon}&run_id=${runId}`);
export const generateReport = (runId: string, bounds?: Bounds) => {
  const params = new URLSearchParams({ run_id: runId });
  if (bounds) {
    params.set("lat_min", String(bounds.lat_min));
    params.set("lat_max", String(bounds.lat_max));
    params.set("lon_min", String(bounds.lon_min));
    params.set("lon_max", String(bounds.lon_max));
  }
  return fetchJSON<Report>(`/explorer/report?${params}`, { method: "POST" });
};

// Types
export interface DatasetInfo {
  row_count: number;
  feature_columns: string[];
  feature_stats: Record<string, { min: number; max: number; mean: number; std: number }>;
  class_distribution: { low_risk: number; high_risk: number };
  geo_bounds: Bounds;
}

export interface Bounds {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface TrainParams {
  n_estimators?: number;
  max_depth?: number | null;
  min_samples_split?: number;
  min_samples_leaf?: number;
  test_size?: number;
  feature_columns?: string[] | null;
}

export interface TrainResult {
  run_id: string;
  metrics: Metrics;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  feature_importance: { feature: string; importance: number }[];
}

export interface Run {
  run_id: string;
  created_at: string;
  params: TrainParams;
  metrics: Metrics;
  feature_names: string[];
  model_path: string;
}

export interface GeocodeResult {
  lat: number;
  lon: number;
  display_name: string;
}

export interface GridResponse {
  cells: GridCell[];
  run_id: string;
}

export interface GridCell {
  lat: number;
  lon: number;
  prediction: number;
  risk_probability: number;
}

export interface CellDetail {
  lat: number;
  lon: number;
  features: Record<string, number>;
  explanation: {
    prediction: number;
    prediction_label: string;
    base_value: number;
    shap_values: { feature: string; value: number; shap_value: number }[];
  };
  summary_text: string;
}

export interface Report {
  region_summary: { name: string; total_cells: number; bounds: Bounds };
  risk_distribution: { high_risk: number; low_risk: number; high_risk_pct: number };
  top_risk_factors: { feature: string; importance: number }[];
  notable_data_points: string[];
  narrative_synthesis: string;
}
```

- [ ] **Step 2: Create DatasetTab component**

```tsx
// frontend/src/components/pipeline/DatasetTab.tsx

import { useEffect, useState } from "react";
import { getDatasetInfo, type DatasetInfo } from "../../lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function DatasetTab() {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    getDatasetInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="text-red-400 p-4">{error}</div>;
  if (!info) return <div className="text-gray-400 p-4">Loading dataset...</div>;

  const distData = [
    { name: "Low Risk", count: info.class_distribution.low_risk, fill: "#22c55e" },
    { name: "High Risk", count: info.class_distribution.high_risk, fill: "#ef4444" },
  ];

  const featureData = Object.entries(info.feature_stats).map(([name, stats]) => ({
    name: name.replace(/_/g, " "),
    mean: stats.mean,
    min: stats.min,
    max: stats.max,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="text-sm text-gray-400">Total Cells</div>
          <div className="text-2xl font-bold">{info.row_count.toLocaleString()}</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="text-sm text-gray-400">Features</div>
          <div className="text-2xl font-bold">{info.feature_columns.length}</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="text-sm text-gray-400">Region</div>
          <div className="text-sm font-mono mt-1">
            {info.geo_bounds.lat_min.toFixed(1)}° to {info.geo_bounds.lat_max.toFixed(1)}°N
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Class Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={distData}>
              <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "#1f2937", border: "1px solid #374151" }} />
              <Bar dataKey="count" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Feature Statistics</h3>
          <div className="overflow-y-auto max-h-52">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-800">
                  <th className="text-left py-1">Feature</th>
                  <th className="text-right py-1">Min</th>
                  <th className="text-right py-1">Mean</th>
                  <th className="text-right py-1">Max</th>
                </tr>
              </thead>
              <tbody>
                {featureData.map((f) => (
                  <tr key={f.name} className="border-b border-gray-800/50">
                    <td className="py-1 text-gray-300">{f.name}</td>
                    <td className="py-1 text-right text-gray-400">{f.min}</td>
                    <td className="py-1 text-right text-gray-300">{f.mean}</td>
                    <td className="py-1 text-right text-gray-400">{f.max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create Pipeline page with tabs**

```tsx
// frontend/src/pages/Pipeline.tsx

import { useState } from "react";
import DatasetTab from "../components/pipeline/DatasetTab";

const TABS = ["Dataset", "Training", "Evaluation"] as const;
type Tab = typeof TABS[number];

export default function Pipeline() {
  const [tab, setTab] = useState<Tab>("Dataset");

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">ML Pipeline</h1>
      <div className="flex gap-1 mb-6 border-b border-gray-800">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab === t
                ? "border-emerald-400 text-emerald-400"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Dataset" && <DatasetTab />}
      {tab === "Training" && <div className="text-gray-400">Training tab — next task</div>}
      {tab === "Evaluation" && <div className="text-gray-400">Evaluation tab — next task</div>}
    </div>
  );
}
```

- [ ] **Step 4: Update App.tsx to use the new Pipeline page**

Replace the placeholder `Pipeline` function in `App.tsx` with:
```tsx
import PipelinePage from "./pages/Pipeline";
```
And update the route:
```tsx
<Route path="/pipeline" element={<PipelinePage />} />
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Pipeline page with Dataset tab"
```

---

### Task 10: Frontend — Pipeline Training Tab

**Files:**
- Create: `frontend/src/components/pipeline/TrainingTab.tsx`
- Modify: `frontend/src/pages/Pipeline.tsx`

- [ ] **Step 1: Create TrainingTab component**

```tsx
// frontend/src/components/pipeline/TrainingTab.tsx

import { useState, useEffect } from "react";
import { trainModel, listRuns, type TrainParams, type Run } from "../../lib/api";

export default function TrainingTab({ onRunComplete }: { onRunComplete?: (runId: string) => void }) {
  const [params, setParams] = useState<TrainParams>({
    n_estimators: 100,
    max_depth: 10,
    min_samples_split: 5,
    min_samples_leaf: 2,
    test_size: 0.2,
  });
  const [training, setTraining] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listRuns().then(setRuns).catch(() => {});
  }, []);

  const handleTrain = async () => {
    setTraining(true);
    setError("");
    try {
      const result = await trainModel(params);
      onRunComplete?.(result.run_id);
      const updated = await listRuns();
      setRuns(updated);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <h3 className="text-sm font-medium text-gray-300 mb-4">Hyperparameters</h3>
        <div className="grid grid-cols-2 gap-4">
          <label className="block">
            <span className="text-xs text-gray-400">n_estimators</span>
            <input
              type="number"
              value={params.n_estimators}
              onChange={(e) => setParams({ ...params, n_estimators: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">max_depth</span>
            <input
              type="number"
              value={params.max_depth ?? ""}
              onChange={(e) => setParams({ ...params, max_depth: e.target.value ? +e.target.value : null })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">min_samples_split</span>
            <input
              type="number"
              value={params.min_samples_split}
              onChange={(e) => setParams({ ...params, min_samples_split: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">min_samples_leaf</span>
            <input
              type="number"
              value={params.min_samples_leaf}
              onChange={(e) => setParams({ ...params, min_samples_leaf: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">test_size</span>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.5"
              value={params.test_size}
              onChange={(e) => setParams({ ...params, test_size: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
        </div>

        <button
          onClick={handleTrain}
          disabled={training}
          className="mt-4 px-6 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium transition-colors"
        >
          {training ? "Training..." : "Train Model"}
        </button>
        {error && <div className="mt-2 text-red-400 text-sm">{error}</div>}
      </div>

      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Run History</h3>
        {runs.length === 0 ? (
          <div className="text-gray-500 text-sm">No runs yet. Train a model to get started.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="text-left py-2">Run ID</th>
                <th className="text-left py-2">Date</th>
                <th className="text-right py-2">Trees</th>
                <th className="text-right py-2">Depth</th>
                <th className="text-right py-2">Accuracy</th>
                <th className="text-right py-2">F1</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                  <td className="py-2 font-mono text-emerald-400">{r.run_id}</td>
                  <td className="py-2 text-gray-400">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="py-2 text-right">{r.params.n_estimators}</td>
                  <td className="py-2 text-right">{r.params.max_depth ?? "None"}</td>
                  <td className="py-2 text-right">{(r.metrics.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-2 text-right">{(r.metrics.f1 * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire TrainingTab into Pipeline page**

Update `Pipeline.tsx` imports and render:
```tsx
import TrainingTab from "../components/pipeline/TrainingTab";
```
Replace the training placeholder:
```tsx
{tab === "Training" && <TrainingTab />}
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: add Pipeline Training tab with hyperparameter config and run history"
```

---

### Task 11: Frontend — Pipeline Evaluation Tab

**Files:**
- Create: `frontend/src/components/pipeline/EvaluationTab.tsx`
- Modify: `frontend/src/pages/Pipeline.tsx`

- [ ] **Step 1: Create EvaluationTab component**

```tsx
// frontend/src/components/pipeline/EvaluationTab.tsx

import { useEffect, useState } from "react";
import { listRuns, getRun, type Run } from "../../lib/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

export default function EvaluationTab() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [compareRun, setCompareRun] = useState<Run | null>(null);

  useEffect(() => {
    listRuns().then((r) => {
      setRuns(r);
      if (r.length > 0) setSelectedRun(r[0]);
    });
  }, []);

  const cm = selectedRun?.metrics.confusion_matrix;
  const featureImportance = selectedRun?.metrics.feature_importance || [];

  return (
    <div className="space-y-6">
      <div className="flex gap-4">
        <label className="block flex-1">
          <span className="text-xs text-gray-400">Select Run</span>
          <select
            value={selectedRun?.run_id || ""}
            onChange={(e) => {
              const r = runs.find((r) => r.run_id === e.target.value);
              setSelectedRun(r || null);
            }}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} — F1: {(r.metrics.f1 * 100).toFixed(1)}%
              </option>
            ))}
          </select>
        </label>
        <label className="block flex-1">
          <span className="text-xs text-gray-400">Compare With</span>
          <select
            value={compareRun?.run_id || ""}
            onChange={(e) => {
              const r = runs.find((r) => r.run_id === e.target.value);
              setCompareRun(r || null);
            }}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            <option value="">None</option>
            {runs
              .filter((r) => r.run_id !== selectedRun?.run_id)
              .map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} — F1: {(r.metrics.f1 * 100).toFixed(1)}%
                </option>
              ))}
          </select>
        </label>
      </div>

      {selectedRun && (
        <div className="grid grid-cols-2 gap-6">
          {/* Metrics summary */}
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <h3 className="text-sm font-medium text-gray-300 mb-3">
              Metrics {compareRun ? "(comparison)" : ""}
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {(["accuracy", "precision", "recall", "f1"] as const).map((m) => (
                <div key={m} className="bg-gray-800 rounded p-3">
                  <div className="text-xs text-gray-400 capitalize">{m}</div>
                  <div className="text-lg font-bold">
                    {(selectedRun.metrics[m] * 100).toFixed(1)}%
                  </div>
                  {compareRun && (
                    <div
                      className={`text-xs ${
                        compareRun.metrics[m] < selectedRun.metrics[m]
                          ? "text-emerald-400"
                          : compareRun.metrics[m] > selectedRun.metrics[m]
                          ? "text-red-400"
                          : "text-gray-500"
                      }`}
                    >
                      vs {(compareRun.metrics[m] * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Confusion matrix */}
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Confusion Matrix</h3>
            {cm && (
              <div className="grid grid-cols-2 gap-2 max-w-xs">
                <div className="bg-emerald-900/30 border border-emerald-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">True Neg</div>
                  <div className="text-xl font-bold text-emerald-400">{cm[0][0]}</div>
                </div>
                <div className="bg-red-900/30 border border-red-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">False Pos</div>
                  <div className="text-xl font-bold text-red-400">{cm[0][1]}</div>
                </div>
                <div className="bg-red-900/30 border border-red-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">False Neg</div>
                  <div className="text-xl font-bold text-red-400">{cm[1][0]}</div>
                </div>
                <div className="bg-emerald-900/30 border border-emerald-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">True Pos</div>
                  <div className="text-xl font-bold text-emerald-400">{cm[1][1]}</div>
                </div>
              </div>
            )}
          </div>

          {/* Feature importance */}
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 col-span-2">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Feature Importance</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={featureImportance} layout="vertical">
                <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={220}
                  tick={{ fill: "#9ca3af", fontSize: 11 }}
                  tickFormatter={(v: string) => v.replace(/_/g, " ")}
                />
                <Tooltip contentStyle={{ background: "#1f2937", border: "1px solid #374151" }} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {featureImportance.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? "#10b981" : i < 3 ? "#34d399" : "#6ee7b7"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Wire EvaluationTab into Pipeline page**

Update imports and replace placeholder:
```tsx
import EvaluationTab from "../components/pipeline/EvaluationTab";
```
```tsx
{tab === "Evaluation" && <EvaluationTab />}
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: add Pipeline Evaluation tab with metrics, confusion matrix, and feature importance"
```

---

### Task 12: Frontend — Explorer Page

**Files:**
- Create: `frontend/src/pages/Explorer.tsx`
- Create: `frontend/src/components/map/RiskMap.tsx`
- Create: `frontend/src/components/map/CellDetailPanel.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create RiskMap component**

```tsx
// frontend/src/components/map/RiskMap.tsx

import { MapContainer, TileLayer, CircleMarker, useMap } from "react-leaflet";
import { type GridCell } from "../../lib/api";
import "leaflet/dist/leaflet.css";

function riskColor(prob: number): string {
  if (prob > 0.7) return "#ef4444";
  if (prob > 0.4) return "#f59e0b";
  return "#22c55e";
}

interface Props {
  cells: GridCell[];
  onCellClick: (cell: GridCell) => void;
  center?: [number, number];
}

function RecenterMap({ center }: { center: [number, number] }) {
  const map = useMap();
  map.setView(center, map.getZoom());
  return null;
}

export default function RiskMap({ cells, onCellClick, center }: Props) {
  const defaultCenter: [number, number] = center || [-12, -61.75];

  return (
    <MapContainer
      center={defaultCenter}
      zoom={8}
      className="h-full w-full rounded-lg"
      style={{ minHeight: "500px" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {center && <RecenterMap center={center} />}
      {cells.map((cell, i) => (
        <CircleMarker
          key={i}
          center={[cell.lat, cell.lon]}
          radius={4}
          pathOptions={{
            fillColor: riskColor(cell.risk_probability),
            fillOpacity: 0.7,
            color: riskColor(cell.risk_probability),
            weight: 1,
          }}
          eventHandlers={{
            click: () => onCellClick(cell),
          }}
        />
      ))}
    </MapContainer>
  );
}
```

- [ ] **Step 2: Create CellDetailPanel component**

```tsx
// frontend/src/components/map/CellDetailPanel.tsx

import { type CellDetail } from "../../lib/api";

interface Props {
  detail: CellDetail | null;
  loading: boolean;
}

export default function CellDetailPanel({ detail, loading }: Props) {
  if (loading) return <div className="text-gray-400 text-sm p-4">Loading cell details...</div>;
  if (!detail) return <div className="text-gray-500 text-sm p-4">Click a cell on the map to see details.</div>;

  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs text-gray-400">Location</div>
        <div className="font-mono text-sm">{detail.lat.toFixed(4)}, {detail.lon.toFixed(4)}</div>
      </div>

      <div>
        <div className={`inline-block px-2 py-1 rounded text-sm font-medium ${
          detail.explanation.prediction === 1
            ? "bg-red-900/50 text-red-300"
            : "bg-emerald-900/50 text-emerald-300"
        }`}>
          {detail.explanation.prediction_label}
        </div>
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-1">Explainability</div>
        <div className="text-sm text-gray-200 bg-gray-800 rounded p-3 font-mono">
          {detail.summary_text}
        </div>
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-2">Features</div>
        <table className="w-full text-sm">
          <tbody>
            {Object.entries(detail.features).map(([key, val]) => (
              <tr key={key} className="border-b border-gray-800/50">
                <td className="py-1 text-gray-400">{key.replace(/_/g, " ")}</td>
                <td className="py-1 text-right font-mono">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-2">SHAP Contributions</div>
        {detail.explanation.shap_values.map((sv) => (
          <div key={sv.feature} className="flex items-center gap-2 text-sm mb-1">
            <span className="text-gray-400 flex-1 truncate">{sv.feature.replace(/_/g, " ")}</span>
            <div className="w-20 h-3 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${sv.shap_value >= 0 ? "bg-red-500" : "bg-emerald-500"}`}
                style={{ width: `${Math.min(Math.abs(sv.shap_value) * 200, 100)}%` }}
              />
            </div>
            <span className={`font-mono text-xs w-14 text-right ${
              sv.shap_value >= 0 ? "text-red-400" : "text-emerald-400"
            }`}>
              {sv.shap_value >= 0 ? "+" : ""}{sv.shap_value.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create Explorer page**

```tsx
// frontend/src/pages/Explorer.tsx

import { useState, useEffect } from "react";
import { listRuns, getGrid, getCellDetail, generateReport, geocodeSearch } from "../lib/api";
import type { Run, GridCell, CellDetail, Report } from "../lib/api";
import RiskMap from "../components/map/RiskMap";
import CellDetailPanel from "../components/map/CellDetailPanel";

export default function Explorer() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [cells, setCells] = useState<GridCell[]>([]);
  const [cellDetail, setCellDetail] = useState<CellDetail | null>(null);
  const [cellLoading, setCellLoading] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [mapCenter, setMapCenter] = useState<[number, number] | undefined>();
  const [error, setError] = useState("");

  useEffect(() => {
    listRuns().then((r) => {
      setRuns(r);
      if (r.length > 0) setSelectedRunId(r[0].run_id);
    });
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    getGrid(selectedRunId).then((g) => setCells(g.cells)).catch((e) => setError(e.message));
  }, [selectedRunId]);

  const handleCellClick = async (cell: GridCell) => {
    setCellLoading(true);
    try {
      const detail = await getCellDetail(cell.lat, cell.lon, selectedRunId);
      setCellDetail(detail);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCellLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    setReportLoading(true);
    try {
      const r = await generateReport(selectedRunId);
      setReport(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setReportLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const result = await geocodeSearch(searchQuery);
      setMapCenter([result.lat, result.lon]);
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="flex h-[calc(100vh-57px)]">
      {/* Sidebar */}
      <div className="w-80 border-r border-gray-800 overflow-y-auto p-4 space-y-4">
        <div>
          <label className="text-xs text-gray-400">Model</label>
          <select
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} (F1: {(r.metrics.f1 * 100).toFixed(1)}%)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-400">Search Location</label>
          <div className="flex gap-2 mt-1">
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g. Porto Velho"
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
            <button
              onClick={handleSearch}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Go
            </button>
          </div>
        </div>

        <div className="border-t border-gray-800 pt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Region Stats</span>
            <span className="text-xs text-gray-500">{cells.length} cells</span>
          </div>
          {cells.length > 0 && (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-emerald-900/30 rounded p-2 text-center">
                <div className="text-xs text-gray-400">Low Risk</div>
                <div className="font-bold text-emerald-400">
                  {cells.filter((c) => c.prediction === 0).length}
                </div>
              </div>
              <div className="bg-red-900/30 rounded p-2 text-center">
                <div className="text-xs text-gray-400">High Risk</div>
                <div className="font-bold text-red-400">
                  {cells.filter((c) => c.prediction === 1).length}
                </div>
              </div>
            </div>
          )}
        </div>

        <button
          onClick={handleGenerateReport}
          disabled={reportLoading || !selectedRunId}
          className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium"
        >
          {reportLoading ? "Generating..." : "Generate Report"}
        </button>

        <div className="border-t border-gray-800 pt-4">
          <CellDetailPanel detail={cellDetail} loading={cellLoading} />
        </div>

        {error && <div className="text-red-400 text-sm">{error}</div>}
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        <RiskMap cells={cells} onCellClick={handleCellClick} center={mapCenter} />
      </div>

      {/* Report overlay */}
      {report && (
        <div className="absolute top-16 right-4 w-96 max-h-[80vh] overflow-y-auto bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-4 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-bold text-sm">Analysis Report</h3>
            <button onClick={() => setReport(null)} className="text-gray-400 hover:text-white text-sm">
              Close
            </button>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1">Region Summary</div>
            <div className="text-sm">{report.region_summary.total_cells} cells analyzed</div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1">Risk Distribution</div>
            <div className="text-sm">
              High risk: {report.risk_distribution.high_risk} ({report.risk_distribution.high_risk_pct}%)
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1">Top Risk Factors</div>
            {report.top_risk_factors.slice(0, 5).map((f) => (
              <div key={f.feature} className="text-sm flex justify-between">
                <span className="text-gray-300">{f.feature.replace(/_/g, " ")}</span>
                <span className="font-mono text-emerald-400">{(f.importance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>

          {report.notable_data_points.length > 0 && (
            <div>
              <div className="text-xs text-gray-400 mb-1">Notable Findings</div>
              <ul className="text-sm space-y-1">
                {report.notable_data_points.map((p, i) => (
                  <li key={i} className="text-yellow-300">• {p}</li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <div className="text-xs text-gray-400 mb-1">Narrative Analysis</div>
            <div className="text-sm text-gray-200 whitespace-pre-wrap">{report.narrative_synthesis}</div>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Update App.tsx**

Replace the inline Explorer function with the import:
```tsx
import ExplorerPage from "./pages/Explorer";
```
Route: `<Route path="/" element={<ExplorerPage />} />`

- [ ] **Step 5: Install leaflet CSS import**

Add to `frontend/src/main.tsx`:
```tsx
import "leaflet/dist/leaflet.css";
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add Explorer page with risk map, cell detail panel, and report generation"
```

---

### Task 13: Integration Testing & Polish

**Files:**
- Modify: various files for fixes found during integration testing

- [ ] **Step 1: Generate the dataset**

```bash
cd backend && venv/Scripts/python -m core.ml.generate_dataset
```

- [ ] **Step 2: Run all backend tests**

```bash
cd backend && venv/Scripts/python -m pytest tests/ -v
```
Expected: All tests pass

- [ ] **Step 3: Start backend and verify API endpoints**

```bash
cd backend && venv/Scripts/python -m uvicorn main:app --port 8000 --reload
```

Test endpoints:
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/pipeline/dataset
curl -X POST http://localhost:8000/api/pipeline/train -H "Content-Type: application/json" -d '{"n_estimators":50,"max_depth":5}'
curl http://localhost:8000/api/pipeline/runs
```

- [ ] **Step 4: Start frontend and verify UI**

```bash
cd frontend && npm run dev
```

Verify:
- Pipeline Dataset tab loads with stats and charts
- Pipeline Training tab: train a model, see it in run history
- Pipeline Evaluation tab: select a run, see confusion matrix and feature importance
- Explorer: select model, see grid on map, click cell for SHAP explanation

- [ ] **Step 5: Fix any issues found during integration**

Debug and fix as needed.

- [ ] **Step 6: Commit fixes**

```bash
git add -A
git commit -m "fix: integration testing fixes"
```

---

### Task 14: README & Documentation

**Files:**
- Create: `README.md`
- Update: `CLAUDE.md`

- [ ] **Step 1: Write README.md**

```markdown
# DeforestAI

A deforestation risk intelligence platform that combines ML-based risk prediction with deterministic explainability and AI-powered report generation.

## The Idea

Deforestation monitoring is critical for conservation, but raw data is hard to act on. DeforestAI makes deforestation risk actionable by:

1. **Predicting risk** — A Random Forest model scores grid cells by deforestation risk based on tabular features (road proximity, elevation, historical loss rate, etc.)
2. **Explaining predictions** — SHAP values provide deterministic, per-cell explanations of why each prediction was made
3. **Generating reports** — Structured reports aggregate findings across a region, with a narrative synthesis layer powered by Claude

## The Thinking

**Architecture choices:**
- **FastAPI + React**: Python backend for natural ML ecosystem access, React frontend for interactive map/dashboard UX
- **Tabular model over CV**: A Random Forest on engineered features is honest, interpretable, and runnable without GPU. The pipeline is model-agnostic — swapping for XGBoost or a CNN requires changing one module.
- **SHAP for explainability**: Deterministic, templated output from real model internals — not LLM-generated explanations. Users see exactly which features drove each prediction.
- **Claude for narrative only**: AI adds value in the hardest part — synthesizing structured data into readable prose. All other report sections are deterministic.

**AI models used:**
- scikit-learn Random Forest (risk classification)
- SHAP TreeExplainer (prediction explainability)
- Claude Sonnet (narrative report synthesis — optional, degrades gracefully without API key)

## Reflections

**What I'd do with more time:**
- Replace synthetic data with real GFW/Hansen API integration
- Add CV-based satellite image classification pipeline (the interfaces are designed for this)
- Implement CI/CD training pipeline automation (scheduled retraining, model promotion)
- Add time-series analysis — track risk changes over multiple model runs
- Multi-region support with per-client model management
- User authentication and saved reports

**Limitations:**
- Dataset is synthetic (realistic distributions, but not real observations)
- Single region (Rondônia) — extending requires data pipeline work
- No real-time data ingestion
- SHAP computation is per-request — would need caching at scale

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- (Optional) Anthropic API key for narrative report generation

### Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix:
source venv/bin/activate

pip install -r requirements.txt
python -m core.ml.generate_dataset
uvicorn main:app --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Environment Variables
```bash
# Optional — report narrative synthesis works without it
export ANTHROPIC_API_KEY=your-key-here
```

### Quick Start
1. Start backend and frontend (see above)
2. Go to Pipeline → Training → click "Train Model"
3. Go to Explorer → select the model from dropdown → see risk map
4. Click any cell to see SHAP explanation
5. Click "Generate Report" for structured analysis
```

- [ ] **Step 2: Update CLAUDE.md with actual architecture**

Update the architecture section in CLAUDE.md to reflect what was actually built.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: add README with project overview, reflections, and setup instructions"
```

---

### Task 15: Final Verification

- [ ] **Step 1: Clean start test**

```bash
# Kill any running servers
# Fresh backend
cd backend && venv/Scripts/python -m core.ml.generate_dataset
cd backend && venv/Scripts/python -m pytest tests/ -v
cd backend && venv/Scripts/python -m uvicorn main:app --port 8000 &

# Fresh frontend
cd frontend && npm install && npm run dev &
```

- [ ] **Step 2: End-to-end walkthrough**

1. Open http://localhost:5173
2. Pipeline → Dataset tab: verify stats display
3. Pipeline → Training: train model with defaults, verify run appears in history
4. Pipeline → Evaluation: select run, verify confusion matrix and feature importance
5. Explorer: select trained model, verify map renders with colored cells
6. Explorer: click a cell, verify SHAP explanation appears
7. Explorer: click Generate Report, verify structured report

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final verification pass"
```
