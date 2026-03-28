import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


class FeatureValidationError(ValueError):
    """вызывается, когда тело запроса не удовлетворяет схеме фичей модели"""


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    if needed_columns is not None:
        unknown = [column for column in needed_columns if column not in FEATURE_COLUMNS]
        if unknown:
            raise FeatureValidationError(
                f'Unsupported feature(s) required by model: {", ".join(unknown)}'
            )
        columns = needed_columns
    else:
        columns = FEATURE_COLUMNS

    missing = [
        column for column in columns if getattr(req, column.replace('.', '_')) is None
    ]
    if missing:
        raise FeatureValidationError(
            f'Missing required feature(s) for active model: {", ".join(missing)}'
        )

    row = [getattr(req, column.replace('.', '_')) for column in columns]
    return pd.DataFrame([row], columns=columns)
