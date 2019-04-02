from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from scipy.io import loadmat, savemat
import numpy as np


def model(input_dim: int=4096) -> Sequential:
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
    model = model_from_json(open(json_path).read())
    return model


def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    for i, layer in enumerate(model.layers):
        weights = dict[str(i)]
        layer.set_weights(weights)
    return model


def conv_dict(dict2):
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


def save_model(model, json_path, weight_path):
    with open(json_path, 'w') as f:
        f.write(model.to_json())

    dict = {}
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
    savemat(weight_path, dict)
