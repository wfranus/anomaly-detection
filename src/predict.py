import os
from argparse import ArgumentParser
from scipy.io import savemat
from src.data_loader import load_features_from_file
from src.model import load_model, load_weights


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='pretrained')
    parser.add_argument('-o', '--output_path', default='results')
    parser.add_argument('-ns', '--segments', type=int, default=32)
    parser.add_argument('-df', '--dim_features', type=int, default=4096)
    parser.add_argument('data', default='data', help='test data directory')

    args = parser.parse_args()

    assert os.path.exists(args.data)
    test_files = sorted(os.listdir(args.data))
    test_paths = [os.path.join(args.data, f) for f in test_files]

    assert os.path.exists(args.model_path)
    mil_model = load_model(os.path.join(args.model_path, 'model.json'))
    load_weights(mil_model, os.path.join(args.model_path, 'weights.mat'))

    if len(test_files):
        os.makedirs(args.output_path, exist_ok=True)

    for i, tp in enumerate(test_paths):
        print(f'Predicting anomaly scores for: {tp}...')
        features = load_features_from_file(tp, args.segments, args.dim_features)
        y_pred = mil_model.predict_on_batch(features)  # anomaly score for each seg.
        out_path = os.path.join(args.output_path, f'{test_files[i][:-4]}.mat')
        savemat(out_path, dict(y_pred=y_pred))
