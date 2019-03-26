import numpy as np
# np.random.seed(42)
# from tensorflow import set_random_seed
# set_random_seed(42)

import os
from argparse import ArgumentParser
from datetime import datetime
from keras.optimizers import Adagrad
from scipy.io import savemat

from src.model import model, save_model
from src.data_loader import load_video_data
from src.loss import custom_loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=20000)
    parser.add_argument('-b', '--batch_size', type=int, default=60)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-s', '--save_path', default='pretrained')
    parser.add_argument('-ns', '--segments', type=int, default=32)
    parser.add_argument('-df', '--dim_features', type=int, default=4096)
    parser.add_argument('data', default='data', help='train data directory')
    args = parser.parse_args()

    abnorm_path = os.path.join(args.data, 'abnorm')
    norm_path = os.path.join(args.data, 'norm')
    assert os.path.exists(abnorm_path) and os.path.exists(norm_path)
    os.makedirs(args.save_path, exist_ok=True)
    model_path = os.path.join(args.save_path, 'model.json')
    weights_path = os.path.join(args.save_path, 'weights.mat')

    mil_model = model(input_dim=args.dim_features)
    optimizer = Adagrad(lr=args.learning_rate, epsilon=1e-08)
    mil_model.compile(optimizer=optimizer,
                      loss=custom_loss(n_bags=args.batch_size,
                                       n_seg=args.segments))

    loss_graph = []
    time_before = datetime.now()

    for it in range(args.iterations):
        inputs, targets = load_video_data(abnorm_path, norm_path,
                                          batch_size=args.batch_size,
                                          n_seg=args.segments,
                                          feat_dim=args.dim_features)
        batch_loss = mil_model.train_on_batch(inputs, targets)
        loss_graph = np.hstack((loss_graph, batch_loss))

        if it % 20 == 1:
            print(f'These iteration={it}) took: {datetime.now() - time_before}, with loss of {batch_loss}')
            iteration_path = os.path.join(args.save_path, f'Iterations_graph_{it}.mat')
            savemat(iteration_path, dict(loss_graph=loss_graph))
        if it % 1000 == 0:
            it_weights_path = os.path.join(args.save_path, f'weightsAnomalyL1L2_{it}.mat')
            save_model(mil_model, model_path, it_weights_path)

    save_model(mil_model, model_path, weights_path)
