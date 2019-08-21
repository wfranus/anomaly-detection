import os
import pytest

from keras.models import Sequential
from numpy.testing import assert_raises, assert_array_equal
from scipy.io import loadmat

from src.model import model, load_model, save_model, load_weights, conv_dict


MODEL_PATH = "tests/fixtures/model.json"
WEIGHTS_PATH = "pretrained/weights_L1L2.mat"


def test_create_model():
    keras_model = model()
    assert isinstance(keras_model, Sequential)


def test_load_model():
    loaded = load_model(MODEL_PATH)
    assert isinstance(loaded, Sequential)


@pytest.mark.skipif(not os.path.isfile(WEIGHTS_PATH),
                    reason='File with weights not found.')
def test_load_weights():
    keras_model = model()
    weights_before = keras_model.get_weights()

    load_weights(keras_model, WEIGHTS_PATH)

    weights_after = keras_model.get_weights()

    # shapes should match, but contents should be different
    assert len(weights_before) == len(weights_after)

    for i in range(len(weights_before)):
        assert weights_before[i].shape == weights_after[i].shape
        assert_raises(AssertionError, assert_array_equal,
                      weights_before[i], weights_after[i])


def test_save_weights():
    keras_model = model()

    save_model(keras_model, "/tmp/model.json", "/tmp/weights.mat")

    assert open("/tmp/model.json").read() == keras_model.to_json()
    loaded_weights = conv_dict(loadmat("/tmp/weights.mat"))
    for i, layer in enumerate(keras_model.layers):
        weights = layer.get_weights()
        if len(weights):
            assert_array_equal(weights[0], loaded_weights[str(i)][0])
            assert_array_equal(weights[1], loaded_weights[str(i)][1])
