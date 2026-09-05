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
