import os
from argparse import ArgumentParser
import numpy as np
from scipy.io import savemat
from src.data_loader import load_all_features_from_dir
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
    parser.add_argument('data', default='data', help='Test data directory.')

    args = parser.parse_args()

    assert os.path.exists(args.data)
    test_files = sorted(os.listdir(args.data))
    norm_videos = [f for f in test_files if 'Normal' in f]
    abnorm_videos = [f for f in test_files if f not in norm_videos]
    test_files = norm_videos + abnorm_videos

    assert os.path.exists(args.model_path)
    mil_model = load_model(os.path.join(args.model_path, 'model.json'))
    load_weights(mil_model, os.path.join(args.model_path, 'weights_L1L2.mat'))

    print('Loading all test data...')
    norm_inputs, abnorm_inputs = load_all_features_from_dir(
            args.data, n_seg=args.segments, feat_dim=args.dim_features)
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
        savemat(out_path, dict(y_pred=y_pred))
