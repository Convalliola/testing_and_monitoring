"""
Мониторинг дрифта через Evidently
Накапливает входящие данные (фичи и предсказания модели) в памяти, затем
периодически строит DataDrift отчёт и загружает его на сервер Evidently
"""

import asyncio
import logging
import threading

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

EVIDENTLY_URL = 'http://158.160.2.37:8000/'
REFERENCE_SIZE = 100 #  минимум записей для формирования baseline
CURRENT_MAX = 1000 # максимальный размер текущего окна (старые записи удаляются)
REPORT_INTERVAL = 60  #  интервал между запусками отчёта, секунды

logger = logging.getLogger(__name__)


class DriftBuffer:
    """Потокобезопасный двухоконный аккумулятор данных для детекции дрифта"""

    def __init__(
        self,
        reference_size: int = REFERENCE_SIZE,
        current_max: int = CURRENT_MAX,
    ) -> None:
        self._lock = threading.Lock()
        self._reference: list[dict] = []
        self._current: list[dict] = []
        self._reference_size = reference_size
        self._current_max = current_max


    # public API

    @property
    def reference_ready(self) -> bool:
        with self._lock:
            return len(self._reference) >= self._reference_size

    def add(self, features: dict, prediction: int, probability: float) -> None:
        """добавляет одно наблюдение (фичи + выход модели) в буфер"""
        record = {**features, 'prediction': prediction, 'probability': probability}
        with self._lock:
            if len(self._reference) < self._reference_size:
                self._reference.append(record)
            else:
                self._current.append(record)
                if len(self._current) > self._current_max:
                    self._current.pop(0)

    def snapshot(self) -> tuple[list[dict], list[dict]]:
        """возвращает независимые копии (reference, current)"""
        with self._lock:
            return list(self._reference), list(self._current)

    def rotate(self) -> None:
        """продвигает текущее окно в эталонное, чистит текущее"""
        with self._lock:
            if self._current:
                self._reference = self._current[-self._reference_size:]
                self._current = []

    def reset(self) -> None:
        """сбрасывает оба окна (вызывается при обновлении модели)"""
        with self._lock:
            self._reference = []
            self._current = []


# Синглтон на уровне модуля, общий для всего приложения
DRIFT_BUFFER = DriftBuffer()


async def drift_monitoring_loop(
    buffer: DriftBuffer,
    project_id: str,
    interval: int = REPORT_INTERVAL,
) -> None:
    """
    Фоновая корутина, строит и загружает отчёт о дрифте каждые [interval] секунд

    Регистрируется при старте сервиса
    asyncio.ensure_future(drift_monitoring_loop(DRIFT_BUFFER, project_id))
    """
    while True:
        await asyncio.sleep(interval)
        try:
            reference, current = buffer.snapshot()

            if len(reference) < REFERENCE_SIZE or len(current) < REFERENCE_SIZE:
                logger.info(
                    'Отчёт о дрифте пропущен — reference=%d current=%d (нужно по %d)',
                    len(reference),
                    len(current),
                    REFERENCE_SIZE,
                )
                continue

            ref_df = pd.DataFrame(reference)
            cur_df = pd.DataFrame(current)

            report = Report(metrics=[DataDriftPreset()])
            result = report.run(reference_data=ref_df, current_data=cur_df)

            workspace = RemoteWorkspace(EVIDENTLY_URL)
            workspace.add_run(project_id, result)

            buffer.rotate()
            logger.info('Отчёт о дрифте загружен (project=%s)', project_id)

        except Exception:  # noqa: BLE001
            logger.exception('Ошибка мониторинга дрифта')
