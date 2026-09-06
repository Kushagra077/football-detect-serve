"""app/main.py's FastAPI endpoints, exercised end-to-end through the real
lifespan/batching machinery but with a FakeBackend swapped in for
app.main.build_backend - so no real model weights or ML deps are ever touched.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from tests.conftest import FakeBackend, tiny_jpeg_bytes


@pytest.fixture
def client(monkeypatch):
    def fake_build_backend(**kwargs):
        return FakeBackend(
            weights=kwargs.get("weights", "fake"),
            imgsz=kwargs.get("imgsz", 640),
            conf=kwargs.get("conf", 0.25),
            iou=kwargs.get("iou", 0.45),
            max_det=kwargs.get("max_det", 300),
        )

    monkeypatch.setattr(main_module, "build_backend", fake_build_backend)
    with TestClient(main_module.app) as c:
        yield c


def test_healthz_reports_warm_and_backends(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["warm"] is True
    assert set(body["backends"]) == set(main_module.SETTINGS.models)


def test_healthz_503_when_no_backends_loaded(client):
    original = main_module.STATE["registry"]
    main_module.STATE["registry"] = {}
    try:
        resp = client.get("/healthz")
        assert resp.status_code == 503
    finally:
        main_module.STATE["registry"] = original


def test_predict_unknown_backend_400(client):
    resp = client.post(
        "/predict",
        params={"backend": "does-not-exist"},
        files={"file": ("f.jpg", tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "unknown backend" in resp.json()["detail"]


def test_predict_missing_body_400(client):
    resp = client.post("/predict")
    assert resp.status_code == 400


def test_predict_happy_path(client):
    resp = client.post(
        "/predict",
        files={"file": ("f.jpg", tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detections"] == []
    assert body["num_detections"] == 0
    assert body["batched"] is True


def test_predict_batch_false_bypasses_queue(client):
    resp = client.post(
        "/predict",
        params={"batch": "false"},
        files={"file": ("f.jpg", tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["batched"] is False
    assert resp.json()["batch_size"] == 1


def test_predict_conf_override_is_call_arg_not_shared_state(client):
    """Guards against the fixed bug: a ?conf= override used to be written onto
    the shared backend object (backend.conf = ...), so it could leak into a
    concurrent request or get reset out from under one still in flight. It
    must now reach predict() as a call argument only, leaving the backend's
    own default untouched.
    """
    backend = main_module.STATE["registry"][main_module.STATE["default"]]["backend"]
    baseline_conf = backend.conf

    resp = client.post(
        "/predict",
        data={"conf": "0.9"},
        files={"file": ("f.jpg", tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200

    assert backend.call_options[-1] == (0.9, None, None)
    assert backend.conf == baseline_conf  # never mutated


def test_predict_bad_backend_label_is_bounded_not_raw_input(client):
    """Guards against unbounded Prometheus cardinality: an invalid ?backend=
    used to be recorded as a metric label verbatim (app/main.py's bname), so a
    client could mint unlimited time series just by varying the query string.
    Every rejected name must now collapse to one fixed sentinel label.
    """
    from app import metrics as metrics_module

    garbage_names = ["nope-1", "nope-2", "!!!weird??", "a" * 200]
    for name in garbage_names:
        resp = client.post(
            "/predict",
            params={"backend": name},
            files={"file": ("f.jpg", tiny_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 400

    samples = metrics_module.REQUESTS.collect()[0].samples
    backend_labels = {
        s.labels["backend"]
        for s in samples
        if s.labels.get("endpoint") == "/predict" and s.labels.get("status") == "400"
    }
    # The registry is a process-wide singleton shared across the whole test
    # session, so other /predict(...400) calls may have already recorded a
    # *real* backend name here (e.g. a missing-body 400 on a valid backend) -
    # that's legitimate. What must never appear is one of the garbage
    # strings themselves, and "invalid" must be the sentinel actually used.
    assert "invalid" in backend_labels
    assert backend_labels.isdisjoint(garbage_names)


def test_predict_not_warm_returns_503(client):
    main_module.STATE["warm"] = False
    try:
        resp = client.post(
            "/predict",
            files={"file": ("f.jpg", tiny_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 503
    finally:
        main_module.STATE["warm"] = True
