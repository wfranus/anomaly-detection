import numpy as np
import tensorflow as tf
# np.random.seed(42)
# from tensorflow import set_random_seed
# set_random_seed(42)

import os
from argparse import ArgumentParser
from datetime import datetime
from keras.optimizers import Adagrad
from tensorflow.summary import FileWriter

from src.model import create_model, save_model
from src.data_loader import load_features_from_dir
from src.loss import custom_loss


if __name__ == '__main__':
    """Train the MIL model.

    Model is trained on 3D features already extracted from videos
    (use prepare_C3D_features.py for feature extraction).
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=20000,
                        help='Number of iterations of MIL model training.')
    parser.add_argument('-b', '--batch_size', type=int, default=60,
                        help='Number of videos (bags in MIL) used during    '
                             'one iteration of model training.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimizer.')
    parser.add_argument('-ns', '--segments', type=int, default=32,
                        help='Number of segments (instances in MIL) '
                             'extracted from each video.')
    parser.add_argument('-df', '--dim_features', type=int, default=4096,
                        help='Dimensionality of video features extracted '
                             'with C3D.')
    parser.add_argument('-s', '--save_path', default='pretrained',
                        help='Where to save trained MIL model.')
    parser.add_argument('data', default='data/mil/train',
                        help='Directory with training data.')
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise FileNotFoundError(args.data)

    os.makedirs(args.save_path, exist_ok=True)
    model_path = os.path.join(args.save_path, 'model.json')
    weights_path = os.path.join(args.save_path, 'weights_L1L2.mat')

    # create file writer to save logs after each iteration
    file_writer = FileWriter(os.path.join(args.save_path, 'logs'),
                             max_queue=1000,
                             flush_secs=10)

    sess = tf.Session()
    with sess.as_default():
        mil_model = create_model(input_dim=args.dim_features)
        optimizer = Adagrad(lr=args.learning_rate, epsilon=1e-08)
        mil_model.compile(optimizer=optimizer,
                          loss=custom_loss(n_bags=args.batch_size,
                                           n_seg=args.segments))

        # load all data in advance to speed up training
        print("Loading all training data..")
        norm_inputs, abnorm_inputs = load_features_from_dir(
                args.data, n_seg=args.segments, feat_dim=args.dim_features)
        print("Data loaded. Training started..")

        # Segments of anomaly video are labeled 1
        # segments of normal video are labeled 0.
        # Here, 1s and 0s are stacked in this order to match order
        # of stacked inputs before shuffling (see loop below).
        targets_size = args.batch_size * args.segments
        targets = np.array([np.ones(targets_size//2, dtype='uint8'),
                            np.zeros(targets_size//2, dtype='uint8')])
        targets = targets.reshape((args.batch_size, args.segments))

        loss_graph = []
        time_before = datetime.now()
        for it in range(args.iterations):
            # randomly choose batch examples
            abnorm_indices = np.random.choice(len(abnorm_inputs),
                                              args.batch_size//2,
                                              replace=False)
            norm_indices = np.random.choice(len(norm_inputs),
                                            args.batch_size//2,
                                            replace=False)
            # stacked array has 3 dims (batch_size, segments, dim_features)
            batch_inputs = np.vstack([abnorm_inputs[abnorm_indices],
                                      norm_inputs[norm_indices]])

            # shuffle inputs over 1st dim (shuffle videos in batch);
            # note: segments of single video CANNOT be shuffled (because of
            # temporal smoothness term in loss function).
            permuted_indices = np.random.permutation(args.batch_size)
            batch_inputs = batch_inputs[permuted_indices]
            batch_targets = targets[permuted_indices]

            # reshape inputs to 2D and targets to 1D
            batch_inputs = batch_inputs.reshape((args.batch_size*args.segments,
                                                 args.dim_features))
            batch_targets = batch_targets.flatten()

            batch_loss = mil_model.train_on_batch(batch_inputs, batch_targets)

            # save loss in logs
            summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                         simple_value=batch_loss)])
            file_writer.add_summary(summary)

            if it % 20 == 1:
                print(f'These iteration={it}) took: {datetime.now() - time_before},'
                      f' with loss of {batch_loss}')
            if it % 1000 == 0:
                it_weights_path = os.path.join(args.save_path,
                                               f'weightsAnomalyL1L2_{it}.mat')
                save_model(mil_model, model_path, it_weights_path)

        save_model(mil_model, model_path, weights_path)
