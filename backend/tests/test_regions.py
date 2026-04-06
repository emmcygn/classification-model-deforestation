import pytest
from core.ml.dataset import load_dataset, list_regions, prepare_features

def test_list_regions():
    regions = list_regions()
    assert len(regions) > 0
    for r in regions:
        assert "name" in r
        assert "cell_count" in r
        assert r["cell_count"] > 0

def test_load_by_region():
    regions = list_regions()
    for r in regions:
        df = load_dataset(region=r["name"])
        assert len(df) == r["cell_count"]
        assert "lat" in df.columns
        assert "high_risk" in df.columns

def test_load_all_regions():
    df = load_dataset()
    assert len(df) > 0
    regions = list_regions()
    total = sum(r["cell_count"] for r in regions)
    assert len(df) == total

def test_prepare_features_from_region():
    df = load_dataset(region=list_regions()[0]["name"])
    X, y, names = prepare_features(df)
    assert X.shape[0] == len(df)
    assert len(y) == len(df)
