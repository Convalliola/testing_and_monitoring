import pytest

from ml_service.features import FeatureValidationError, to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_selects_needed_columns(sample_payload):
    req = PredictRequest(**sample_payload)
    df = to_dataframe(req, needed_columns=['age', 'education.num'])
    assert list(df.columns) == ['age', 'education.num']
    assert int(df.iloc[0]['age']) == 39
    assert int(df.iloc[0]['education.num']) == 13


def test_to_dataframe_raises_on_missing_required_feature(sample_payload):
    sample_payload.pop('age')
    req = PredictRequest(**sample_payload)
    with pytest.raises(FeatureValidationError, match='Missing required feature'):
        to_dataframe(req, needed_columns=['age', 'workclass'])


def test_to_dataframe_raises_on_unknown_needed_feature(sample_payload):
    req = PredictRequest(**sample_payload)
    with pytest.raises(FeatureValidationError, match='Unsupported feature'):
        to_dataframe(req, needed_columns=['age', 'unknown_feature'])
