import os
from argparse import ArgumentParser
import numpy as np
from scipy.io import savemat
from src.data_loader import load_features_from_files
from src.data_loader import normal_abnormal_split
from src.model import load_model, load_weights


if __name__ == '__main__':
    """Predict anomaly scores for video segments.

    It uses the pretrained MIL model created by train.py.
    """
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='pretrained',
                        help='Path to the saved MIL model.')
    parser.add_argument('-o', '--output_path', default='results',
                        help='Where to store predictions.')
    parser.add_argument('-ns', '--segments', type=int, default=32,
                        help='Number of segments in MIL. Must be the same '
                             'as during training.')
    parser.add_argument('-df', '--dim_features', type=int, default=4096,
                        help='Dimensionality of video features.')
    parser.add_argument('-r', '--ratio', type=float,
                        help='Randomly choose subset of test videos such that '
                             'percentage of anomaly videos in chosen subset is '
                             'nearly equal to given ratio value. By default '
                             'all videos from data directory are considered.')
    parser.add_argument('data', default='data', help='Test data directory.')

    args = parser.parse_args()

    assert os.path.exists(args.data)
    test_files = sorted(os.listdir(args.data))
    norm_videos, abnorm_videos = normal_abnormal_split(test_files, args.ratio)

    # simulate anomaly to normal class ratio given by user,
    # maximizing total test set size at the same time
    if args.ratio:
        print(f'Balancing data to the class ratio defined by user: {args.ratio}')
        print(f'Randomly chosen anomaly videos: {len(abnorm_videos)}')
        print(f'Randomly chosen normal videos: {len(norm_videos)}')
        real_ratio = np.round(len(abnorm_videos)/
                              (len(abnorm_videos)+len(norm_videos)),
                              2)
        print(f'Percentage of anomalies in total videos: {real_ratio}')

    test_files = norm_videos + abnorm_videos

    assert os.path.exists(args.model_path)
    mil_model = load_model(os.path.join(args.model_path, 'model.json'))
    load_weights(mil_model, os.path.join(args.model_path, 'weights_L1L2.mat'))

    print('Loading all test data...')
    norm_inputs = load_features_from_files(norm_videos, args.data,
                                           args.segments, args.dim_features)
    abnorm_inputs = load_features_from_files(abnorm_videos, args.data,
                                             args.segments, args.dim_features)
    # TODO: refactor
    if (norm_inputs is not None) and (abnorm_inputs is not None):
        inputs = np.vstack([norm_inputs, abnorm_inputs])
    elif norm_inputs is not None:
        inputs = norm_inputs
    else:
        inputs = abnorm_inputs
    print('Test data loaded. Start prediction...')

    if len(test_files):
        os.makedirs(args.output_path, exist_ok=True)

    for i, tp in enumerate(test_files):
        print(f'Predicting anomaly scores for: {tp}...')
        y_pred = mil_model.predict_on_batch(inputs[i])  # anomaly score for each seg.
        out_path = os.path.join(args.output_path, f'{test_files[i][:-4]}.mat')
        savemat(out_path, dict(y_pred=y_pred))  # TODO: matlab really?
