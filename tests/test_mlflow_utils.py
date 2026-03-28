import pytest

import ml_service.mlflow_utils as mlflow_utils
from ml_service.mlflow_utils import ModelLoadError, get_model_uri, load_model


def test_get_model_uri_rejects_empty_run_id():
    with pytest.raises(ModelLoadError, match='non-empty'):
        get_model_uri('')


def test_load_model_success(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mlflow_utils, '_load_sklearn_model', lambda _uri: sentinel)
    model = load_model(run_id='abc123')
    assert model is sentinel


def test_load_model_wraps_errors(monkeypatch):
    def _raise(_uri):
        raise Exception('mlflow failure')

    monkeypatch.setattr(mlflow_utils, '_load_sklearn_model', _raise)
    with pytest.raises(ModelLoadError, match='Failed to load model'):
        load_model(run_id='bad-run')
