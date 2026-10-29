from fastapi.testclient import TestClient

from backend.api import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "api_activa"
    assert "X-Response-Time-ms" in r.headers


def test_schema_lists_features():
    client = TestClient(app)
    r = client.get("/schema")
    assert "loc" in r.json()["features"]
    assert "negocio" in r.json()


def test_train_forbidden_without_admin():
    client = TestClient(app)
    r = client.post("/train", headers={"X-Role": "operativo"})
    assert r.status_code == 403


def test_predict_without_model_is_404(sample_df):
    client = TestClient(app)
    r = client.post("/predict", json={"module_id": "m1", "metricas": {"loc": 10}})
    assert r.status_code in (404, 422, 503)
