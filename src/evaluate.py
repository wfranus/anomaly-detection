import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from argparse import ArgumentParser
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve
)
from utils.prepare_C3D_features import count_frames

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TODO: parametrize or load from config?
# number of frames in single clip used in C3D feature extraction
C3D_FPC = 16
# fully connected layer of C3D net used to get features
C3D_FC = 'fc6-1'
# number of segments (instances) used in MIL model
N_SEG = 32
# threshold for anomaly score above which predicted class is "anomaly"
SCORE_THRESH = 0.5


def evaluate():
    parser = ArgumentParser()
    parser.add_argument('--video_dir', default='data/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos',  # noqa
                        help='Directory with test videos. Required to compute '
                             'frame counts for each video.')
    parser.add_argument('--temp_ann', default='data/Temporal_Anomaly_Annotation.txt',
                        help='File with temporal annotations of test videos.')
    parser.add_argument('--c3d_features', default='data/c3d/test',
                        help='Directory with C3D features. Needed to count '
                             'number of generated features for each video '
                             '(number of files in each subdirectory).')
    parser.add_argument('-s', '--scores', default='results',
                        help='Directory with predictions (scores).')
    parser.add_argument('-o,', '--out', default='eval_results',
                        help='Where to save evaluation results.')
    args = parser.parse_args()

    # check provided paths
    if not os.path.isdir(args.video_dir):
        logger.error(f'Invalid path to video directory: {args.video_dir}')
        sys.exit()
    if not os.path.isfile(args.temp_ann):
        logger.error(f'Invalid path to annotations file: {args.temp_ann}')
        sys.exit()
    if not os.path.isdir(args.c3d_features):
        logger.error(f'Invalid path to C3D features directory: {args.c3d_features}')
        sys.exit()
    if not os.path.isdir(args.scores):
        logger.error(f'Invalid path to predictions directory: {args.scores}')
        sys.exit()

    pred_file_names = sorted(os.listdir(args.scores))
    pred_file_paths = [os.path.join(args.scores, f) for f in pred_file_names]

    # load annotations
    annotations = pd.read_csv(args.temp_ann, r'\s+', header=None,
                              names=['name', 'class', 'sf1', 'ef1', 'sf2', 'ef2'])

    # predictions (scores) and ground truth for all frames from all videos
    all_predictions, all_ground_truth = None, None

    for i, pred_file in enumerate(pred_file_paths):
        # name of the sample video without extension
        video_name = pred_file_names[i].rstrip('.mat')

        # annotations for video
        ann_row = annotations[annotations.name.str.contains(video_name)]

        # path to video relative to video_dir
        vid_rel_path = ann_row.iloc[0]['name']

        # count total frames in video
        video_file = os.path.join(args.video_dir, vid_rel_path)
        n_frames, _ = count_frames(video_file, accurate=False)

        # count frames used in C3D feature extraction (possibly subset of total)
        c3d_feat_dir = os.path.join(args.c3d_features, video_name)
        n_c3d_features = len([f for f in os.listdir(c3d_feat_dir)
                              if f.endswith(C3D_FC)])
        n_frames_c3d = n_c3d_features * C3D_FPC

        # load predictions for frames used in C3D feature extraction
        y_pred = loadmat(pred_file).get('y_pred')

        if y_pred is None:
            logger.error(f'could not load data from: {pred_file}')
            continue

        # predictions for all video frames
        detection_scores = np.zeros((1, n_frames), dtype='float32')

        # assign predictions computed for video segments to each frame.
        breakpoints = np.linspace(0, n_frames_c3d, N_SEG + 1)
        breakpoints = np.floor(breakpoints + 0.5).astype('int').tolist()

        for j in range(len(breakpoints) - 1):
            # ss - starting index, ee - end (after last) index
            ss, ee = breakpoints[j], breakpoints[j + 1]
            # print(f'ss: {ss}, ee: {ee}, j: {j}, y_pred[j] {y_pred[j]}')

            if ee <= ss:
                detection_scores[0, ss] = y_pred[j]
            else:
                detection_scores[0, ss:ee] = y_pred[j]

        # frames not used during C3D feature extraction are assigned score
        # of the last frame that was used in feature extraction
        if n_frames_c3d < n_frames:
            logger.debug(f'n_frames_c3d: {n_frames_c3d}, n_frames: {n_frames}, '
                         f'vid={video_name}')
            detection_scores[0, n_frames_c3d:n_frames] = \
                detection_scores[0, n_frames_c3d-1]

        # read ground truth labels from annotation file
        # anomaly frames are labeled 1, normal 0
        ground_truth = np.zeros((1, n_frames), dtype='int32')
        sf1, ef1 = ann_row.iloc[0]['sf1'], ann_row.iloc[0]['ef1']
        sf2, ef2 = ann_row.iloc[0]['sf2'], ann_row.iloc[0]['ef2']
        if sf1 != -1 and ef1 != -1:
            ground_truth[0, sf1:ef1] = 1
        if sf2 != -1 and ef2 != -1:
            ground_truth[0, sf2:ef2] = 1

        if i == 0:
            all_predictions = detection_scores
            all_ground_truth = ground_truth
        else:
            all_predictions = np.hstack((all_predictions, detection_scores))
            all_ground_truth = np.hstack((all_ground_truth, ground_truth))

    if all_predictions is not None and all_ground_truth is not None:
        all_predictions = all_predictions.transpose()
        all_ground_truth = all_ground_truth.transpose()

        # calculate statistics
        fpr, tpr, thresholds = roc_curve(all_ground_truth, all_predictions)
        roc_auc = auc(fpr, tpr)
        y_predicted = np.where(all_predictions >= SCORE_THRESH, 1, 0)
        acc = accuracy_score(all_ground_truth, y_predicted)
        report = classification_report(all_ground_truth, y_predicted,
                                       target_names=['normal', 'anomaly'])

        # calculate false positive rate for normal videos' frames
        normal_predicted = y_predicted[np.where(all_ground_truth == 0)]
        normal_ground_truth = np.zeros_like(normal_predicted)
        cm = confusion_matrix(normal_ground_truth, normal_predicted)
        FPR_norm = cm[0][1] / normal_predicted.shape[0]  # TODO: in denominator just take total number of normal videos

        # create text report
        out_report = [
            f"Accuracy: {acc:.3f}",
            f"FPR for normal videos: {FPR_norm:.3f}",
            f"AUC: {roc_auc:.4f}\n",
            report
        ]
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'report.txt'), 'w+') as f:
            f.write('\n'.join(out_report))

        # create ROC plot
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.out, "roc.png"))
        plt.show()


if __name__ == '__main__':
    evaluate()
