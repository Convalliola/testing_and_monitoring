import mlflow

from ml_service import config


class ModelLoadError(RuntimeError):
    """Вызывается, когда модель не удаётся загрузить из MLflow."""


def configure_mlflow() -> None:
    uri = config.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def get_model_uri(run_id: str) -> str:
    if not run_id or not run_id.strip():
        raise ModelLoadError('run_id must be a non-empty string')
    return f'runs:/{run_id}/model'


def _load_sklearn_model(model_uri: str):
    return mlflow.sklearn.load_model(model_uri)


def load_model(model_uri: str = None, run_id: str = None) -> mlflow.pyfunc.PyFuncModel:
    """
    Downloads artifacts locally (if needed) and loads model as an MLflow PyFunc model.
    """
    try:
        if not model_uri:
            model_uri = get_model_uri(run_id)
        return _load_sklearn_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        src = model_uri if model_uri else f'run_id={run_id}'
        raise ModelLoadError(f'Failed to load model from {src}') from exc

