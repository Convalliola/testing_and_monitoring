from fastapi.testclient import TestClient

import ml_service.app as service_app
from ml_service.model import ModelData


class FakePipelineModel:
    feature_names_in_ = ['age', 'workclass']

    def predict_proba(self, _df):
        return [[0.1, 0.9]]


def _install_fake_model_loader(monkeypatch):
    def fake_set(run_id: str):
        if run_id == 'bad-run':
            raise service_app.ModelLoadError('bad run id')
        with service_app.MODEL.lock:
            service_app.MODEL.data = ModelData(model=FakePipelineModel(), run_id=run_id)

    monkeypatch.setattr(service_app, 'configure_mlflow', lambda: None)
    monkeypatch.setattr(service_app.config, 'default_run_id', lambda: 'startup-run')
    monkeypatch.setattr(service_app.MODEL, 'set', fake_set)


def test_predict_happy_path(monkeypatch, sample_payload):
    _install_fake_model_loader(monkeypatch)
    app = service_app.create_app()
    with TestClient(app) as client:
        response = client.post('/predict', json=sample_payload)
        assert response.status_code == 200
        body = response.json()
        assert body['prediction'] == 1
        assert body['probability'] == 0.9


def test_predict_validation_error_for_missing_required_feature(monkeypatch, sample_payload):
    _install_fake_model_loader(monkeypatch)
    app = service_app.create_app()
    sample_payload.pop('workclass')
    with TestClient(app) as client:
        response = client.post('/predict', json=sample_payload)
        assert response.status_code == 400
        assert 'Missing required feature' in response.json()['detail']


def test_predict_returns_503_when_model_is_not_loaded(monkeypatch, sample_payload):
    _install_fake_model_loader(monkeypatch)
    app = service_app.create_app()
    with TestClient(app) as client:
        with service_app.MODEL.lock:
            service_app.MODEL.data = ModelData(model=None, run_id=None)
        response = client.post('/predict', json=sample_payload)
        assert response.status_code == 503


def test_predict_payload_type_validation(monkeypatch, sample_payload):
    _install_fake_model_loader(monkeypatch)
    app = service_app.create_app()
    sample_payload['age'] = 'invalid-int'
    with TestClient(app) as client:
        response = client.post('/predict', json=sample_payload)
        assert response.status_code == 422


def test_update_model_bad_run_id(monkeypatch):
    _install_fake_model_loader(monkeypatch)
    app = service_app.create_app()
    with TestClient(app) as client:
        response = client.post('/updateModel', json={'run_id': 'bad-run'})
        assert response.status_code == 400


def test_metrics_endpoint(monkeypatch):
    _install_fake_model_loader(monkeypatch)
    app = service_app.create_app()
    with TestClient(app) as client:
        response = client.get('/metrics')
        assert response.status_code == 200
        assert 'ml_service_http_requests_total' in response.text
