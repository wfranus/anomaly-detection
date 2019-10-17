import os
from typing import Optional, Sequence, Tuple, List
import numpy as np


def load_video_data_batch(features_dir, batch_size=60,
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


def load_features_from_files(paths, features_dir: str,
                             n_seg: int = 32, feat_dim: int = 4096)\
        -> Optional[np.ndarray]:
    if not paths:
        return None

    features = []
    for i, file in enumerate(paths):
        abs_path = os.path.join(features_dir, file)
        loaded_features = load_features_from_file(abs_path, n_seg, feat_dim)
        features.append(loaded_features)

    return np.stack(features)


def load_features_from_dir(features_dir: str, abnorm_ratio: float = None,
                           n_seg: int = 32, feat_dim: int = 4096) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    all_files = sorted(os.listdir(features_dir))
    norm_videos, abnorm_videos = normal_abnormal_split(all_files, abnorm_ratio)

    norm_features = load_features_from_files(norm_videos, features_dir,
                                             n_seg, feat_dim)
    abnorm_features = load_features_from_files(abnorm_videos, features_dir,
                                               n_seg, feat_dim)

    return norm_features, abnorm_features


def calc_new_lenghts_with_ratio(ratio: float, norm_len: int,
                                anomalies_len: int) \
        -> Tuple[int, int]:
    def calc_anomalies(ratio, norm_len):
        return - ratio * norm_len / (ratio - 1)

    def calc_normal(ratio, anomalies_len):
        return anomalies_len * (1 - ratio) / ratio

    x_anomalies_len = calc_anomalies(ratio, norm_len)

    if x_anomalies_len > anomalies_len:
        x_normal_len = calc_normal(ratio, anomalies_len)
        return round(x_normal_len), round(calc_anomalies(ratio, x_normal_len))

    return round(calc_normal(ratio, x_anomalies_len)), round(x_anomalies_len)


def normal_abnormal_split(video_list: Sequence[str],
                          abnorm_ratio: float = None) \
        -> Tuple[List[str], List[str]]:
    norm_videos = [f for f in video_list if 'Normal' in f]
    abnorm_videos = [f for f in video_list if f not in norm_videos]

    if abnorm_ratio is not None:
        if abnorm_ratio == 0.0:
            abnorm_videos = []
        elif abnorm_ratio == 1.0:
            norm_videos = []
        else:
            new_norm_len, new_abnorm_len = calc_new_lenghts_with_ratio(
                    abnorm_ratio, len(norm_videos), len(abnorm_videos))
            abnorm_videos = list(np.random.choice(abnorm_videos, new_abnorm_len,
                                                  False))
            norm_videos = list(np.random.choice(norm_videos, new_norm_len,
                                                False))

    return norm_videos, abnorm_videos
