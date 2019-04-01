import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from argparse import ArgumentParser
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_curve
)
from utils.prepare_C3D_features import count_frames

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# number of frames in single clip used in C3D feature extraction
C3D_FPC = 16
# fully connected layer of C3D net used to get features
C3D_FC = 'fc6-1'
# number of segments (instances) used in MIL model
N_SEG = 32


def evaluate():
    parser = ArgumentParser()
    parser.add_argument('--video_dir', default='data/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos')
    parser.add_argument('--temp_ann', default='data/Temporal_Anomaly_Annotation.txt')
    parser.add_argument('--c3d_features', default='data/c3d/test',
                        help='directory with C3D features')
    parser.add_argument('--scores', default='results',
                        help='directory with predictions (scores)')
    parser.add_argument('--out', default='eval_results')
    args = parser.parse_args()

    # check provided paths
    if not os.path.isdir(args.video_dir):
        logger.error(f'Invalid path to video directory: {args.video_dir}')
        sys.exit()
    if not os.path.isdir(args.temp_ann):
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
    annotations = np.genfromtxt(args.temp_ann, delimiter='  ', autostrip=True,
                                dtype=[('name', 'U100'), ('class', 'U50'),
                                       ('sf1', 'i'), ('ef1', 'i'),
                                       ('sf2', 'i'), ('ef2', 'i')])

    # predictions (scores) and ground truth for all frames from all videos
    all_predictions, all_ground_truth = None, None

    for i, pred_file in enumerate(pred_file_paths):
        # name of the sample video without extension
        video_name = pred_file_names[i].rstrip('.mat')

        # load predictions
        y_pred = loadmat(pred_file)

        # count total frames in video
        video_file = os.path.join(args.video_dir, video_name + '.mp4')
        n_frames, _ = count_frames(video_file, accurate=False)

        # count frames used in C3D feature extraction
        c3d_feat_dir = os.path.join(args.c3d_features, video_name)
        n_c3d_features = len([f for f in os.listdir(c3d_feat_dir) if f.endswith(C3D_FC)])
        n_frames_c3d = n_c3d_features * C3D_FPC

        # computer breakpoints in matlab 1-based indexing for consistence
        # with original matlab implementation of this function
        breakpoints = np.linspace(1, n_frames_c3d, N_SEG + 1)
        breakpoints = np.floor(breakpoints + 0.5).astype('int').tolist()
        # convert from 1-based to 0-based indexing
        breakpoints = np.subtract(breakpoints, 1)

        detection_scores = np.zeros((1, n_frames), dtype='float32')

        for j in range(len(breakpoints) - 1):
            # ss - starting index, ee - end (after last) index
            ss, ee = breakpoints[j], breakpoints[j + 1]

            if ee < ss:
                detection_scores[ss * C3D_FPC:(ss + 1) * C3D_FPC] = y_pred[j]
            else:
                detection_scores[ss * C3D_FPC:(ee + 1) * C3D_FPC] = y_pred[j]

        # frames not used during C3D feature extraction are assigned score
        # of the last frame that was used in feature extraction
        if n_frames_c3d < n_frames:
            detection_scores[n_frames_c3d:n_frames] = detection_scores[n_frames_c3d-1]

        # read ground truth labels from annotation file
        # anomaly frames are labeled 1, normal 0
        ground_truth = np.zeros((1, n_frames), dtype='int32')
        ann_row = annotations[np.where(annotations['name'] == video_name)][0]
        if ann_row['sf1'] != -1 and ann_row['ef1'] != -1:
            ground_truth[ann_row['sf1']:ann_row['ef1']] = 1
        if ann_row['sf2'] != -1 and ann_row['ef2'] != -1:
            ground_truth[ann_row['sf2']:ann_row['ef2']] = 1

        if i == 0:
            all_predictions = detection_scores
            all_ground_truth = ground_truth
        else:
            all_predictions = np.hstack((all_predictions, detection_scores))
            all_ground_truth = np.hstack((all_ground_truth, ground_truth))

    if all_predictions and all_ground_truth:
        fpr, tpr, thresholds = roc_curve(all_ground_truth, all_predictions)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(all_ground_truth, all_predictions)
        report = classification_report(all_ground_truth, all_predictions,
                                       target_names=['normal', 'anomaly'])
        # create text report
        out_report = [
            f"Accuracy: {acc:.3f}\n",
            report
        ]
        with open(os.path.join(args.out, 'report.txt'), 'w+') as f:
            f.write('\n'.join(out_report))

        # create ROC plot
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()

        os.makedirs(args.out, exist_ok=True)
        plt.savefig("roc.png")
        

if __name__ == '__main__':
    evaluate()
