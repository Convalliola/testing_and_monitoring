import asyncio
from contextlib import asynccontextmanager
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from ml_service import config
from ml_service.drift import DRIFT_BUFFER, drift_monitoring_loop
from ml_service.features import FeatureValidationError, to_dataframe
from ml_service.metrics import (
    ACTIVE_MODEL_FEATURES_COUNT,
    ACTIVE_MODEL_INFO,
    ERROR_COUNTER,
    FEATURE_VALUE_GAUGE,
    INFERENCE_DURATION,
    MODEL_UPDATES_TOTAL,
    MODEL_UPDATE_DURATION,
    PREDICTION_CLASS_COUNTER,
    PREDICTION_PROBABILITY,
    PREPROCESS_DURATION,
    REQUEST_COUNTER,
    REQUEST_DURATION,
    observe_duration,
    render_metrics,
)
from ml_service.mlflow_utils import ModelLoadError, configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


MODEL = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application

    При старте загружает начальную модель из MLflow и запускает фоновую
    корутину мониторинга дрифта
    """
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    ACTIVE_MODEL_INFO.clear()
    ACTIVE_MODEL_INFO.labels(
        run_id=run_id,
        model_type=MODEL.model_type or 'unknown',
    ).set(1)
    ACTIVE_MODEL_FEATURES_COUNT.set(len(MODEL.features))

    project_id = config.evidently_project_id()
    if project_id:
        asyncio.ensure_future(drift_monitoring_loop(DRIFT_BUFFER, project_id))

    yield


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def metrics_middleware(request: Request, call_next):
        endpoint = request.url.path
        method = request.method
        with observe_duration(REQUEST_DURATION, endpoint=endpoint, method=method):
            response = await call_next(request)
        REQUEST_COUNTER.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(response.status_code),
        ).inc()
        return response

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.get('/metrics')
    def metrics() -> Response:
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model_state = MODEL.get()
        model = model_state.model
        if model is None:
            ERROR_COUNTER.labels(endpoint='/predict', error_type='model_not_loaded').inc()
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        try:
            with observe_duration(PREPROCESS_DURATION):
                df = to_dataframe(request, needed_columns=MODEL.features)
        except FeatureValidationError as exc:
            ERROR_COUNTER.labels(endpoint='/predict', error_type='invalid_features').inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for feature in df.columns:
            value = df.iloc[0][feature]
            if isinstance(value, (int, float, np.number)):
                FEATURE_VALUE_GAUGE.labels(feature=feature).set(float(value))

        try:
            with observe_duration(INFERENCE_DURATION):
                probability = float(model.predict_proba(df)[0][1])
        except Exception as exc:  # noqa: BLE001
            ERROR_COUNTER.labels(endpoint='/predict', error_type='inference_error').inc()
            raise HTTPException(status_code=500, detail='Model inference failed') from exc

        prediction = int(probability >= 0.5)
        PREDICTION_PROBABILITY.observe(probability)
        PREDICTION_CLASS_COUNTER.labels(prediction=str(prediction)).inc()

        DRIFT_BUFFER.add(
            features=df.iloc[0].to_dict(),
            prediction=prediction,
            probability=probability,
        )

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        start = time.perf_counter()
        try:
            MODEL.set(run_id=run_id)
        except ModelLoadError as exc:
            MODEL_UPDATE_DURATION.labels(status='failed').observe(time.perf_counter() - start)
            MODEL_UPDATES_TOTAL.labels(status='failed').inc()
            ERROR_COUNTER.labels(endpoint='/updateModel', error_type='invalid_run_id').inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            MODEL_UPDATE_DURATION.labels(status='failed').observe(time.perf_counter() - start)
            MODEL_UPDATES_TOTAL.labels(status='failed').inc()
            ERROR_COUNTER.labels(endpoint='/updateModel', error_type='internal_error').inc()
            raise HTTPException(status_code=500, detail='Model update failed') from exc

        MODEL_UPDATE_DURATION.labels(status='success').observe(time.perf_counter() - start)
        MODEL_UPDATES_TOTAL.labels(status='success').inc()
        ACTIVE_MODEL_INFO.clear()
        ACTIVE_MODEL_INFO.labels(
            run_id=run_id,
            model_type=MODEL.model_type or 'unknown',
        ).set(1)
        ACTIVE_MODEL_FEATURES_COUNT.set(len(MODEL.features))
        DRIFT_BUFFER.reset()
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
