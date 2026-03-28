import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

REQUEST_COUNTER = Counter(
    'ml_service_http_requests_total',
    'Суммарное число HTTP-запросов, обработанных сервисом',
    ['endpoint', 'method', 'status_code'],
)
REQUEST_DURATION = Histogram(
    'ml_service_http_request_duration_seconds',
    'Время обработки HTTP-запроса, секунды',
    ['endpoint', 'method'],
    buckets=LATENCY_BUCKETS,
)
ERROR_COUNTER = Counter(
    'ml_service_errors_total',
    'Суммарное число обработанных ошибок',
    ['endpoint', 'error_type'],
)
PREPROCESS_DURATION = Histogram(
    'ml_service_preprocess_duration_seconds',
    'Время предобработки входных данных, секунды',
    buckets=LATENCY_BUCKETS,
)
INFERENCE_DURATION = Histogram(
    'ml_service_inference_duration_seconds',
    'Время инференса модели, секунды',
    buckets=LATENCY_BUCKETS,
)
MODEL_UPDATE_DURATION = Histogram(
    'ml_service_model_update_duration_seconds',
    'Время обновления модели, секунды',
    ['status'],
    buckets=LATENCY_BUCKETS,
)
PREDICTION_PROBABILITY = Histogram(
    'ml_service_prediction_probability',
    'Распределение вероятностей предсказания класса 1',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
PREDICTION_CLASS_COUNTER = Counter(
    'ml_service_prediction_class_total',
    'Число предсказаний по классам',
    ['prediction'],
)
MODEL_UPDATES_TOTAL = Counter(
    'ml_service_model_updates_total',
    'Число попыток обновления модели',
    ['status'],
)
ACTIVE_MODEL_INFO = Gauge(
    'ml_service_active_model_info',
    'Метаданные активной модели (константное значение 1)',
    ['run_id', 'model_type'],
)
ACTIVE_MODEL_FEATURES_COUNT = Gauge(
    'ml_service_active_model_features_count',
    'Количество фичей, необходимых активной модели',
)
FEATURE_VALUE_GAUGE = Gauge(
    'ml_service_feature_last_value',
    'Последнее числовое значение входной фичи',
    ['feature'],
)


@contextmanager
def observe_duration(histogram: Histogram, **labels: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
