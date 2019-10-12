"""
This script extracts C3D features from videos
and converts them to match MIL input data format.

Based on Python2 script from official C3D repo:
https://github.com/facebook/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/extract_C3D_feature.py
"""
import array
import cv2
import logging
import math
import os
import re
import subprocess
import sys
import numpy as np
import yaml
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from typing import Sequence, Tuple
from yaml import Loader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_trained_model(trained_model):
    """Check if trained_model is there. Otherwise, download"""

    if os.path.isfile(trained_model):
        logger.info(f"trained_model={trained_model} found. Good to go!")
    else:
        download_cmd = [
            "wget",
            "-O",
            trained_model,
            "https://www.dropbox.com/s/vr8ckp0pxgbldhs/conv3d_deepnetA_sport1m_iter_1900000?dl=0",
        ]

        logger.info("Download Sports1m pre-trained model: {download_cmd}")
        return_code = subprocess.call(download_cmd)

        if return_code != 0:
            logger.error("Downloading of pretrained model failed. Check!")
            sys.exit(-10)


def count_frames(path: str, accurate: bool = False) -> Tuple[int, float]:
    """Return number of frames in video and frames per second rate.

    Args:
        path:     path to the video file
        accurate: if set, frames will be counted one-by-one which is
                  slower, but always gives accurate result. Otherwise
                  frame count is read from video metadata which sometimes
                  can be inaccurate or missing.

    Return:
        A tuple (number of frames in video file, frames per second)
    """
    # FIXME: The real use of opencv here is for quick peering of frame count
    # FIXME: and FPS. It should not be required to have opencv installed
    # FIXME: when manual method is chosen by the user.
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        logger.error(f"video={path} can not be opened.")
        sys.exit(-6)

    def manual_count(video):
        count = 0
        while video.grab():
            count += 1
        return count

    def is_cv2():
        return cv2.__version__.startswith('2.')

    if accurate:
        total = manual_count(video)
    else:
        try:
            if is_cv2():
                total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            else:
                total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            logger.debug(f'missing frame count in video metadata: {path}')
            total = manual_count(video)

    if is_cv2():
        fps = float(video.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        fps = float(video.get(cv2.CAP_PROP_FPS))

    return total, fps


def run_C3D_extraction(c3d_root, feature_prototxt: str,
                       trained_model: str, ofile: str,
                       gpu_id: int = 0, batch_size: int = 50,
                       feature_layer: str = 'fc6-1', ) -> int:
    """Extract C3D features by running extract_image_features.bin binary"""
    extract_bin = os.path.join(
        c3d_root,
        "build/tools/extract_image_features.bin"
    )

    if not os.path.isfile(extract_bin):
        logger.error("Build facebook/C3D first, or make sure c3d_root is "
                     "correct")
        sys.exit(-9)

    # compute number of mini batches based on the number of clips
    with open(ofile, 'r') as f:
        clip_count = 0
        for _ in f:
            clip_count += 1
    n_batches = math.ceil(clip_count / batch_size)

    feature_extraction_cmd = [
        extract_bin,
        feature_prototxt,
        trained_model,
        str(gpu_id),
        str(batch_size),
        str(n_batches),
        ofile,
        feature_layer,
    ]

    logger.info(f"Running C3D feature extraction: {feature_extraction_cmd}")
    return_code = subprocess.call(feature_extraction_cmd)

    return return_code


def create_input_prototxt(input_dir: str,
                          input_files: Sequence[str] = None,
                          out_dir: str = os.path.curdir,
                          out_filename: str = 'input.txt',
                          start_frame: int = 0,
                          stride: int = 16,
                          max_clips: int = None,
                          force_accurate_frames: bool = True):
    """Create input prototxt file for C3D feature extractor.

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
        out_dir:     path to directory where to store the generated file
        out_filename:name of the generated file
        start_frame: starting frame of the first clip. If it is greater than
                     the total number of video frames, the video file will be
                     skipped
        stride:      number of frames in a single clip (16 is default for C3D)
        max_clips:   maximum number of clips to be generated from each video.
                     If None, all clips will be used.
        force_accurate_frames: whether to use slower but always accurate
                     method for frame counting.
    """
    if not input_files:
        input_files = sorted(os.listdir(input_dir))
    input_files = [os.path.join(input_dir, f) for f in input_files]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_filename), 'w') as f:
        for file in input_files:
            n_frames, _ = count_frames(file, accurate=force_accurate_frames)
            logger.debug(f'frames: {n_frames} - {file}')

            if start_frame > n_frames:
                logger.info(f'start_frame > n_frames for: {file}. Skipping..')
                continue

            n_clips = int((n_frames-start_frame)/stride)
            if max_clips is None or max_clips <= 0 or max_clips > n_clips:
                _max_clips = n_clips
            else:
                _max_clips = max_clips

            s_frame = start_frame
            dummy_label = 0
            for _ in range(_max_clips):
                # write line describing single clip consisting of stride frames,
                # syntax is: <path_to_file> <starting_frame_indexed_from_0> <label>
                # int label is not used during feature extraction, but must be present
                f.write(f'{file} {s_frame} {dummy_label}\n')
                s_frame += stride


def create_output_prefix_file(input_prototxt: str,
                              out_dir: str = None,
                              out_filename: str = 'output_prefix.txt',
                              out_features_dir: str = 'output/c3d'):
    """Create output prefix prototxt file for C3D feature extractor.

    Args:
        input_prototxt:  path to an existing input prototxt file or to
                         directory with this file. If path to directory is,
                         given, then the file should be named input.txt
        out_dir:         path to directory where to store the generated file.
                         This should be set if this is not the same directory
                         as for input file.
        out_filename:    name of the generated file.
        out_features_dir:path to directory where C3D features will be stored.
                         It is common for all input videos, but clips related
                         to the same video will be saved in subdirectory
                         under feat_output_dir. The names of subdirectories
                         are inferred from input_prototxt and they are
                         the names of input videos.
    """
    if os.path.isdir(input_prototxt):
        input_prototxt = os.path.join(input_prototxt, 'input.txt')

    with open(input_prototxt, 'r') as f:
        input_lines = f.readlines()

    out_features_dir.rstrip('/')
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.abspath(os.path.dirname(os.fspath(input_prototxt)))

    with open(os.path.join(out_dir, out_filename), 'w') as f:
        for iline in input_lines:
            file_name, sframe, _ = iline.split(' ')
            file_name = os.path.splitext(os.path.basename(file_name))[0]
            f.write(f'{out_features_dir}/{file_name}/{int(sframe):06}\n')

            # create directory for c3d features (important!)
            os.makedirs(os.path.join(out_features_dir, file_name), exist_ok=True)


def create_network_prototxt(out_file: str, c3d_root: str, src_file: str,
                            mean_file: str = None, batch_size: int = 50):
    """Create prototxt file with C3D network configuration.

    Default configuration of the C3D pretrained on sport1m dataset will be used.
    Only batch size and paths to input config files can be modified.

    Args:
        out_file:   path to the output file
        c3d_root:   path to the root of C3D v1.0 project
        src_file:   path to the existing file with description of input data
        mean_file:  if set, path to custom binaryproto file describing the model.
                    By default an appropriate file from C3D examples will be taken.
        batch_size: batch size of the model. Batch size of 50 should be fine
                    for 6GB VRAM. Reduce or increase this parameter depend on
                    your GPU.
    """
    c3d_feat_extr_dir = os.path.join(
        c3d_root,
        'examples',
        'c3d_feature_extraction',
    )
    if not mean_file:
        mean_file = os.path.join(
            c3d_feat_extr_dir,
            'sport1m_train16_128_mean.binaryproto'
        )

    if not os.path.isfile(mean_file):
        logger.error(f"mean cube file={mean_file} does not exist.")
        sys.exit(-8)

    template_prototxt = os.path.join(
        c3d_feat_extr_dir,
        'prototxt',
        'c3d_sport1m_feature_extractor_video.prototxt'
    )

    with open(template_prototxt, 'r') as f:
        config = f.read()

    # adjust source, mean_file and batch_size parameters of data layer
    config = re.sub(r'(source:) ".*"', r'\1 "' + str(src_file) + '"', config)
    config = re.sub(r'(mean_file:) ".*"', r'\1 "' + str(mean_file) + '"', config)
    config = re.sub(r'(batch_size:) \d+', r'\1 ' + str(batch_size), config)

    with open(out_file, 'w') as f:
        f.write(config)


def load_array_from_blob(file_path: str, feat_dim: int = 4096) -> np.ndarray:
    with open(file_path, 'rb') as f:
        s = array.array("i")
        s.fromfile(f, 5)  # 5 32-bit integers: num, channel, length, height, width
        data = array.array("f")
        data.fromfile(f, feat_dim)
        data = np.array(data.tolist())

    return data


def split_c3d_features_to_segments(feat_dir: str, feat_dim: int = 4096,
                                   n_seg: int = 32, fc: str = 'fc6-1',
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

    # FIXME: breakpoints initialized like this give EXACTLY the same
    # FIXME: results as MATLAB implementation, but should be refactored
    # breakpoints contains indices of all_features indicating breakpoints
    # for n_seg segments. Breakpoints are computed on 1-based indexing,
    # following matlab implementation of this function.
    breakpoints = np.linspace(1, all_features.shape[0], n_seg + 1)
    # np.rint implementation is inconsistent with matlab round function!
    # breakpoints = np.rint(breakpoints).astype('int').tolist()
    breakpoints = np.floor(breakpoints + 0.5).astype('int').tolist()
    # convert from 1-based to 0-based indexing
    breakpoints = np.subtract(breakpoints, 1)

    seg_features = np.zeros((n_seg, feat_dim), dtype='float32')
    for i in range(len(breakpoints) - 1):
        # ss - starting index, ee - end (after last) index
        # it matlab we had to subtract 1 from ee, because end index
        # points to last element (inclusively) in matlab
        ss, ee = breakpoints[i], breakpoints[i+1]

        if ee <= ss:
            tmp_vec = all_features[ss, :]
        else:
            tmp_vec = np.mean(all_features[ss:ee, :], axis=0)

        # apply L2 normalization on the newly created vector
        seg_features[i, :] = tmp_vec/np.linalg.norm(tmp_vec)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_name = out_name or os.path.basename(feat_dir)

        if store_binary:
            np.save(os.path.join(out_dir, f'{out_name}.npy'), seg_features)
        else:
            np.savetxt(os.path.join(out_dir, f'{out_name}.txt'),
                       seg_features, fmt='%.6f', delimiter=' ')

    return seg_features


def split_features_from_dir(c3d_out_dir, out_mil, n_seg: int = 32) -> None:
    """Split all C3D features in c3d_out_dir into n_seg segments."""
    logger.info("Converting extracted features to MIL data format.")
    feat_dirs = [f.path for f in os.scandir(c3d_out_dir) if f.is_dir()]
    feat_dirs = sorted(feat_dirs)

    for dir in feat_dirs:
        split_c3d_features_to_segments(feat_dir=dir, out_dir=out_mil,
                                       n_seg=n_seg)


def prepare_C3D_features(args):
    """Extract C3D features from videos and convert these features
    into desired number of new features to match MIL input data format.

    Args:
        args: namespace or dictionary-like object with arguments
              (see main function for argument description)
    """
    if not (isinstance(args, SimpleNamespace) or isinstance(args, Namespace)):
        args = SimpleNamespace(**args)

    # read user input file
    with open(args.input_file, 'r') as f:
        test_files = f.read().splitlines()

    # c3d config
    prototxt_dir = '/tmp'
    input_config_file = 'input.txt'
    output_config_file = 'output_prefix.txt'
    feature_extraction_config = 'feature_extraction.prototxt'

    logger.info("Preparing input files for C3D extractor.")
    create_input_prototxt(
        input_dir=args.video_dir,
        input_files=test_files,
        out_dir=prototxt_dir,
        out_filename=input_config_file,
        force_accurate_frames=(not args.fast)
    )
    create_output_prefix_file(
        input_prototxt=prototxt_dir,
        out_dir=prototxt_dir,
        out_filename=output_config_file,
        out_features_dir=args.out_c3d
    )
    create_network_prototxt(
        out_file=os.path.join(prototxt_dir, feature_extraction_config),
        c3d_root=args.c3d_root,
        src_file=os.path.join(prototxt_dir, input_config_file),
        mean_file=None,
        batch_size=args.batch_size
    )
    logger.info("Running C3D extractor...")
    ret_code = run_C3D_extraction(
        args.c3d_root,
        feature_prototxt=os.path.join(prototxt_dir, feature_extraction_config),
        trained_model=args.model,
        ofile=os.path.join(prototxt_dir, output_config_file),
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        feature_layer='fc6-1'
    )
    # ret_code = 0
    if ret_code == 0:
        logger.info("Feature extraction completed!")
        split_features_from_dir(args.out_c3d, args.out_mil, args.num_seg)
    else:
        logger.error("Feature extraction failed!")


def main():
    """Extract C3D features from videos and convert these features
    into desired number of new features to match MIL input data format.

    If you have already extracted C3D features and only want to create
    features for MIL, run with --segment_only flag and pass --out_c3d.

    Before running make sure you have installed C3D model and checked
    configuration in config.yml.
    """
    parser = ArgumentParser()
    parser.add_argument('--num_seg', type=int, default=32,
                        help='Number of new features to be created for each '
                             'video. Must match MIL input data format.')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='If true, try to read frame count from video '
                             'metadata. If false or such information is not '
                             'provided in metadata, count video frames manually '
                             'which always gives accurate result but is slow.')
    parser.add_argument('--video_dir', default='data/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos',  # noqa
                        help='Directory with videos.')
    parser.add_argument('--input_file', default='data/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt',  # noqa
                        help='Text file listing paths to videos for which '
                             'features should be extracted. Paths must be '
                             'relative to "video_dir".')
    parser.add_argument('--out_c3d', default='data/c3d/test',
                        help='Output directory for C3D features.')
    parser.add_argument('--out_mil', default='data/mil/test',
                        help='Output directory for MIL model input data '
                             '(segmented C3D features).')
    parser.add_argument('--segment_only', action='store_true',
                        help='If set, skip C3D extraction and only segment '
                             '3D features. Extracted C3D features must be '
                             'stored in "out_c3d" directory.')
    args = parser.parse_args()

    # fast path for segmentation
    if args.segment_only:
        if not os.path.isdir(args.out_c3d):
            logger.error(f'Directory with C3D features doesn\'t exist: {args.video_dir}')
            sys.exit()

        return split_features_from_dir(args.out_c3d, args.out_mil, args.num_seg)

    # check provided paths
    if not os.path.isdir(args.video_dir):
        logger.error(f'Video directory doesn\'t exist: {args.video_dir}')
        sys.exit()
    if not os.path.isfile(args.input_file):
        logger.error(f'Input file doesn\'t exist: {args.input_file}')
        sys.exit()

    # load C3D model configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               'config.yaml')
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    args = Namespace(**vars(args),
                     model=cfg['C3D']['model'],
                     c3d_root=cfg['C3D']['root'],
                     batch_size=cfg['C3D']['batch_size'],
                     gpu=cfg['gpu'])

    # check if pretrained model exists and download if necessary
    check_trained_model(args.model)

    # run pipeline
    return prepare_C3D_features(args)


if __name__ == '__main__':
    main()
