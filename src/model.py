from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from scipy.io import loadmat, savemat


def create_model(input_dim: int = 4096) -> Sequential:
    """Create Keras model used during MIL learning.

    It is a 3-layer perceptron with 512, 32 and 1 neurons accordingly,
    regularized with Dropout layers and L2 regularizers after each
    hidden layer.
    """
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001)))

    return model


def load_model(json_path):
    """Load model (without weights) from json file."""
    model = model_from_json(open(json_path).read())
    return model


def load_weights(model, weight_path):
    """Load weights from matlab files and assign them to model layers."""
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    for i, layer in enumerate(model.layers):
        weights = dict[str(i)]
        layer.set_weights(weights)
    return model


def conv_dict(dict2):
    """Convert dictionary from Matlab-like style to Python style."""
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape[0] == 1:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def save_model(model: Sequential, json_path: str, weights_file_path: str):
    """Save given model and its weights to files.
    Weights are saved in Matlab format.
    All parent directories within `json_path` and `weights_file_path`
    must exist.
    """
    with open(json_path, 'w') as f:
        f.write(model.to_json())

    dict = {}
    for i, layer in enumerate(model.layers):
        dict[str(i)] = layer.get_weights()

    savemat(weights_file_path, dict)
