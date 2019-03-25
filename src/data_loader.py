import os
import numpy as np


def load_video_data(abnorm_path, norm_path, batch_size=60,
                    n_seg=32, feat_dim=4096, verbose=0):
    """Load features of abnormal and normal videos from files."""
    assert batch_size % 2 == 0, 'Batch size must be multiple of 2'
    n_exp = batch_size // 2  # Number of abnormal and normal videos

    abnorm_videos = sorted(os.listdir(abnorm_path))
    norm_videos = sorted(os.listdir(norm_path))

    assert n_exp <= len(abnorm_videos)
    assert n_exp <= len(norm_videos)

    # n_abnorm = 810  # Total number of abnormal videos in Training Dataset.
    # n_norm = 800  # Total number of Normal videos in Training Dataset.

    abnorm_indices = np.random.choice(len(abnorm_videos), n_exp, replace=False)
    norm_indices = np.random.choice(len(norm_videos), n_exp, replace=False)

    batch_videos = [os.path.join(abnorm_path, abnorm_videos[id]) for id in abnorm_indices]
    batch_videos += [os.path.join(norm_path, norm_videos[id]) for id in norm_indices]

    if verbose:
        print("Loading features...")

    # TODO: maybe store C3D in binary format rather than textual for easier loading?
    def load_features_from_file(file_path) -> np.ndarray:
        with open(file_path, "r") as f:
            words = f.read().split()

        num_feat = len(words) / feat_dim
        assert num_feat == n_seg

        vid_features = []
        for feat in range(n_seg):
            feat_row = np.float32(words[feat * feat_dim:feat * feat_dim + feat_dim])
            if feat == 0:
                vid_features = feat_row
            else:
                vid_features = np.vstack((vid_features, feat_row))

        return vid_features

    batch_features = []  # To store C3D features of a batch

    for i, video_path in enumerate(batch_videos):
        vid_features = load_features_from_file(video_path)
        if i == 0:
            batch_features = vid_features
        else:
            batch_features = np.vstack((batch_features, vid_features))

    if verbose:
        print("Features loaded")

    # segments of abnormal videos are labeled 0
    # while segments of normal videos are labeled 1
    targets = np.zeros(n_seg * batch_size, dtype='uint8')
    targets[n_exp * n_seg:] = 1

    return batch_features, targets
