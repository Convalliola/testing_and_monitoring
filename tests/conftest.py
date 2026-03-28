import pytest


class DummyModel:
    def __init__(self, features: list[str]) -> None:
        self.feature_names_in_ = features

    def predict_proba(self, _df):
        return [[0.2, 0.8]]


@pytest.fixture
def sample_payload() -> dict:
    return {
        'age': 39,
        'workclass': 'State-gov',
        'fnlwgt': 77516,
        'education': 'Bachelors',
        'education.num': 13,
        'marital.status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital.gain': 2174,
        'capital.loss': 0,
        'hours.per.week': 40,
        'native.country': 'United-States',
    }
