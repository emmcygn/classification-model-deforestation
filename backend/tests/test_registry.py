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
