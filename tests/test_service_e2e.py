from fastapi.testclient import TestClient

import ml_service.app as service_app
from ml_service.model import ModelData


class E2EModel:
    feature_names_in_ = ['age', 'workclass']

    def predict_proba(self, _df):
        return [[0.8, 0.2]]


def test_service_smoke(monkeypatch, sample_payload):
    def fake_set(run_id: str):
        with service_app.MODEL.lock:
            service_app.MODEL.data = ModelData(model=E2EModel(), run_id=run_id)

    monkeypatch.setattr(service_app, 'configure_mlflow', lambda: None)
    monkeypatch.setattr(service_app.config, 'default_run_id', lambda: 'e2e-run')
    monkeypatch.setattr(service_app.MODEL, 'set', fake_set)

    app = service_app.create_app()
    with TestClient(app) as client:
        health_resp = client.get('/health')
        assert health_resp.status_code == 200
        assert health_resp.json()['run_id'] == 'e2e-run'

        predict_resp = client.post('/predict', json=sample_payload)
        assert predict_resp.status_code == 200
        assert predict_resp.json()['prediction'] == 0

        metrics_resp = client.get('/metrics')
        assert metrics_resp.status_code == 200
        assert 'ml_service_inference_duration_seconds' in metrics_resp.text
