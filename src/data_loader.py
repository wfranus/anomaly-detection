import os
import numpy as np
import pandas as pd


def load_video_data(features_dir, batch_size=60,
                    n_seg=32, feat_dim=4096, verbose=0):
    """Load features of abnormal and normal videos from files."""
    assert batch_size % 2 == 0, 'Batch size must be multiple of 2'
    n_exp = batch_size // 2  # Number of abnormal and normal videos in batch

    all_files = sorted(os.listdir(features_dir))
    norm_videos = [f for f in all_files if 'Normal' in f]
    abnorm_videos = [f for f in all_files if f not in norm_videos]

    assert n_exp <= len(abnorm_videos)
    assert n_exp <= len(norm_videos)

    abnorm_indices = np.random.choice(len(abnorm_videos), n_exp, replace=False)
    norm_indices = np.random.choice(len(norm_videos), n_exp, replace=False)

    batch_videos = [os.path.join(features_dir, abnorm_videos[id])
                    for id in abnorm_indices]
    batch_videos += [os.path.join(features_dir, norm_videos[id])
                     for id in norm_indices]

    if verbose:
        print("Loading features...")

    batch_features = []  # To store C3D features of a batch

    for i, video_path in enumerate(batch_videos):
        vid_features = load_features_from_file(video_path, n_seg, feat_dim)
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


def load_all_features_from_dir(features_dir: str, n_seg: int, feat_dim: int = 4096) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    all_files = sorted(os.listdir(features_dir))
    norm_videos = [f for f in all_files if 'Normal' in f]
    abnorm_videos = [f for f in all_files if f not in norm_videos]

    norm_features = []
    for i, file in enumerate(norm_videos):
        abs_path = os.path.join(features_dir, file)
        norm_features.append(load_features_from_file(abs_path, n_seg, feat_dim))

    norm_features = np.stack(norm_features)

    abnorm_features = []
    for i, file in enumerate(abnorm_videos):
        abs_path = os.path.join(features_dir, file)
        abnorm_features.append(load_features_from_file(abs_path, n_seg, feat_dim))

    abnorm_features = np.stack(abnorm_features)

    norm_targets = np.ones(len(norm_videos), dtype='uint8')
    abnorm_targets = np.zeros(len(abnorm_videos), dtype='uint8')
    return norm_features, abnorm_features, norm_targets, abnorm_targets


def load_features_from_file(file_path: str, n_seg: int, feat_dim: int = 4096) \
        -> np.ndarray:
    """Load 2D array of features from file.

    Args:
        file_path: Path to file
        n_seg: number of segments (# rows)
        feat_dim: number of features for each segment (# cols)

    Returns:
        A numpy array of shape (n_seg, feat_dim) and type float32.
    """
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
