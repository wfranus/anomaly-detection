import array
import cv2
import os
import numpy as np
from typing import Sequence


def count_frames(path: str, accurate: bool = False) -> int:
    """Count number of frames in video file.

    Args:
        path:     path to the video file
        accurate: if set, frames will be counted one-by-one which is
                  slower, but always gives accurate result. Otherwise
                  frame count is read from video metadata which sometimes
                  can be inaccurate or missing.

    Return:
        A number of frames in video file
    """
    video = cv2.VideoCapture(path)
    total = 0

    def manual_count(video):
        count = 0
        while video.grab():
            count += 1
        return count

    if accurate:
        total = manual_count(video)
    else:
        try:
            if any(major in cv2.__version__ for major in ['3.', '4.']):
                total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            elif '2.' in cv2.__version__:
                total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        except:
            total = manual_count(video)

    return total


def create_input_list_video_file(input_dir: str,
                                 input_files: Sequence[str] = None,
                                 output_dir: str = os.path.curdir,
                                 start_frame: int = 0,
                                 stride: int = 16,
                                 max_clips: int = None):
    """Create input_list_video.txt file for C3D feature extractor.

    This text file defines input video clips for C3D feature extractor.
    Each line contains path to the video file, clip starting frame and
    integer label for the clip. Label must be present but is ignored
    by feature extractor. Frames are 0-indexed.

    Args:
        input_dir:   path to directory with video files [.avi, .mov, .mp4]
        input_files: optional, list of paths to video files. This argument
                     should be used if: (1) video files are stored in
                     different directories or (2) only subset of files
                     in input_dir should be processed
        output_dir:  path to directory where to store the generated file
        start_frame: starting frame of the first clip. If it is greater than
                     the total number of video frames, the video file will be
                     skipped
        stride:      number of frames in a single clip (16 is default for C3D)
        max_clips:   maximum number of clips to be generated from each video.
                     If None, all clips will be used.
    """
    if not input_files:
        input_files = sorted(os.listdir(input_dir))
        input_files = [os.path.join(input_dir, f) for f in input_files]

    output_file = 'input_list_video.txt'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as f:
        for file in input_files:
            n_frames = count_frames(file, accurate=True)
            print(f'{n_frames} - {file}')

            if start_frame > n_frames:
                print(f'start_frame > n_frames for: {file}. Skipping..')
                continue

            n_clips = int((n_frames-start_frame)/stride)
            if max_clips is None or max_clips <= 0 or max_clips > n_clips:
                max_clips = n_clips

            s_frame = start_frame
            for _ in range(max_clips):
                # write line describing single clip consisting of stride frames,
                # syntax is: <path_to_file> <starting_frame_indexed_from_0> <label>
                # int label is not used during feature extraction, but must be present
                f.write(f'{file} {s_frame} 0\n')
                s_frame += stride


def create_output_list_video_prefix_file(input_list_file: str,
                                         output_dir: str = os.path.curdir,
                                         feat_output_dir: str = 'output/c3d'):
    """Create output_list_video_prefix.txt file for C3D feature extractor.

    Args:
        input_list_file: path to existing input_list_video.txt file
        output_dir:      path to directory where to store the generated file
        feat_output_dir: path to directory where C3D features will be stored.
                         It is common for all input videos, but clips related
                         to the same video will be saved in subdirectory
                         under feat_output_dir. The name of subdirectory
                         is inferred from input_list_video.txt file and it is
                         the name of input video.
    """
    input_lines = open(input_list_file, 'r').readlines()
    feat_output_dir.rstrip('/')

    output_file = 'output_list_video_prefix.txt'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as f:
        for iline in input_lines:
            file_name, sframe, _ = iline.split(' ')
            file_name = os.path.splitext(os.path.basename(file_name))[0]
            f.write(f'{feat_output_dir}/{file_name}/{int(sframe):06}\n')


def load_array_from_blob(file_path: str, feat_dim: int=4096) -> np.ndarray:
    with open(file_path, 'rb') as f:
        s = array.array("i")
        s.fromfile(f, 5)  # 5 32-bit integers: num, channel, length, height, width
        data = array.array("f")
        data.fromfile(f, feat_dim)
        data = np.array(data.tolist())

    return data


def split_c3d_features_to_segments(feat_dir: str, feat_dim: int=4096,
                                   n_seg: int=32, fc: str = 'fc6-1',
                                   out_dir: str = None,
                                   out_name: str = None,
                                   store_binary: bool = False) \
        -> np.ndarray:
    """Split default C3D video features into n_seg features.

    By default, C3D computes features for each 16-frame video clip.
    Each such feature vector is stored in a separate blob file.
    This function reads features from all such files located in feat_dir,
    and splits them into n_seg features, taking average if needed.
    For each new feature vector L2 normalization is applied.

    Args:
        feat_dir: path to directory with features generated by C3D
                  feature extractor (default format of file names in expected,
                  i.e. <start_frame>.<layer_name>)
        feat_dim: dimensionality of feature vectors
        n_seg:    desired number of new features
        fc:       name of one of the FC layers to extract features from.
                  fc6-1 (default) and fc7-1 are allowed
        out_dir:  if specified, output directory where to save new features
        out_name: name (without extension) for the output file with new features.
                  By default, basename of feat_dir directory will be used
        store_binary: if set, new features will be stored in binary .npy format,
                      by default new features are stored in text file

    Returns:
        A NumPy array [n_seg, feat_dim] with new features
    """
    files = sorted(os.listdir(feat_dir))
    files = [os.path.join(feat_dir, f) for f in files if f.endswith(fc)]

    # all_features contains feature vector for each 16-frame video clip
    all_features = np.zeros((len(files), feat_dim), dtype='float32')
    for i, file in enumerate(files):
        all_features[i, :] = load_array_from_blob(file, feat_dim)
        print(file)

    # split_ranges contains indices of all_features indicating split points
    # for n_seg segments. Split points are computed on 1-based indexing,
    # following matlab implementation of this function.
    split_ranges = np.linspace(1, all_features.shape[0], n_seg + 1)
    # np.rint implementation is inconsistent with matlab round function!
    # split_ranges = np.rint(split_ranges).astype('int').tolist()
    split_ranges = np.floor(split_ranges + 0.5).astype('int').tolist()
    # convert from 1-based to 0-based indexing
    split_ranges = np.subtract(split_ranges, 1)

    seg_features = np.zeros((n_seg, feat_dim), dtype='float32')
    for i in range(len(split_ranges) - 1):
        # ss - starting index, ee - end (after last) index
        # it matlab we had to subtract 1 from ee, because end index
        # points to last element (inclusively) in matlab
        ss, ee = split_ranges[i], split_ranges[i+1]

        if ee <= ss:
            tmp_vec = all_features[ss, :]
        else:
            tmp_vec = np.mean(all_features[ss:ee, :], axis=0)

        seg_features[i, :] = tmp_vec/np.linalg.norm(tmp_vec)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_name = out_name or os.path.basename(os.path.dirname(feat_dir))
        print(out_name)

        if store_binary:
            np.save(os.path.join(out_dir, f'{out_name}.npy'), seg_features)
        else:
            np.savetxt(os.path.join(out_dir, f'{out_name}.txt'),
                       seg_features, fmt='%.6f', delimiter=' ')

    return seg_features
