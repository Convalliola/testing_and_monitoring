from ml_service.model import Model


def test_model_set_and_features(monkeypatch):
    class LocalDummyModel:
        feature_names_in_ = ['age', 'workclass']

    monkeypatch.setattr('ml_service.model.load_model', lambda run_id: LocalDummyModel())

    model = Model()
    model.set('run-1')

    assert model.get().run_id == 'run-1'
    assert model.features == ['age', 'workclass']
    assert model.model_type == 'LocalDummyModel'


def test_model_set_is_atomic_on_failed_update(monkeypatch):
    class OldModel:
        feature_names_in_ = ['age']

    class NewModel:
        feature_names_in_ = ['age', 'workclass']

    calls = {'count': 0}

    def load_model_side_effect(run_id):
        calls['count'] += 1
        if calls['count'] == 1:
            return OldModel()
        raise RuntimeError('boom')

    monkeypatch.setattr('ml_service.model.load_model', load_model_side_effect)

    model = Model()
    model.set('old-run')
    previous = model.get()

    try:
        model.set('new-run')
    except RuntimeError:
        pass

    current = model.get()
    assert current.run_id == previous.run_id
    assert current.model is previous.model
