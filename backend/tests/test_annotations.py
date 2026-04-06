import pytest
import asyncio
from core.ml.annotations import AnnotationStore

@pytest.fixture
def store(tmp_path):
    s = AnnotationStore(tmp_path / "test_annotations.db")
    asyncio.get_event_loop().run_until_complete(s.init())
    return s

def test_save_and_list(store):
    loop = asyncio.get_event_loop()
    aid = loop.run_until_complete(store.save_annotation(
        lat=9.75, lon=118.74, run_id="test1",
        prediction=1, risk_probability=0.85,
        verdict="accept", note="Confirmed logging activity",
    ))
    assert aid is not None

    annotations = loop.run_until_complete(store.list_annotations("test1"))
    assert len(annotations) == 1
    assert annotations[0]["verdict"] == "accept"
    assert annotations[0]["note"] == "Confirmed logging activity"

def test_get_annotation_for_cell(store):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(store.save_annotation(
        lat=10.0, lon=119.0, run_id="test2",
        prediction=0, risk_probability=0.3,
        verdict="reject", note="Known farm clearing",
    ))
    result = loop.run_until_complete(store.get_annotation_for_cell(10.0, 119.0, "test2"))
    assert result is not None
    assert result["verdict"] == "reject"

def test_get_annotation_nonexistent(store):
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(store.get_annotation_for_cell(99.0, 99.0, "none"))
    assert result is None

def test_stats(store):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(store.save_annotation(lat=1.0, lon=1.0, run_id="r1", prediction=1, risk_probability=0.9, verdict="accept"))
    loop.run_until_complete(store.save_annotation(lat=2.0, lon=2.0, run_id="r1", prediction=1, risk_probability=0.8, verdict="accept"))
    loop.run_until_complete(store.save_annotation(lat=3.0, lon=3.0, run_id="r1", prediction=1, risk_probability=0.7, verdict="reject"))
    stats = loop.run_until_complete(store.get_stats("r1"))
    assert stats["accepted"] == 2
    assert stats["rejected"] == 1
    assert stats["total"] == 3
