# anomaly-detection
Anomaly detection in surveillance videos using deep multiple instance learning (MIL).

This is an implementation of [this paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf):
```
Real-world Anomaly Detection in Surveillance Videos
Sultani, Waqas and Chen, Chen and Shah, Mubarak,
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018
```
This repo is an updated, refactored version of the [original implementation](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018), rewritten in Python3 only (original code was in Matlab and Python2).

## Requirements
* Python 3.6+
* C3D v1.0
* Keras 2.2.4 with Tensorflow 1.9.0 backend (tested on these versions only)

## Installation
Follow instructions in `INSTALL.md` to install C3D and its dependencies.
To install Python dependencies run:  
```pip install -r requirements.txt```

To verify that installation was successful run tests:  
```pytest tests```

## Running
### Using pretrained MIL model
* To extract C3D features from test videos and convert them to format accepted
by the MIL model adjust `config.yml` and run:  
```python utils/prepare_C3D_features --fast --out_c3d data/c3d/test --out_mil data/mil/test --input_file data/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt```

    Extracted raw C3D features will be stored in directory pointed by `--out_c3d`.
    Segmented features, in the format expected by the MIL model will be stored in
    directory pointed by `--out_mil`.

* To predict anomaly scores for test videos using the pretrained MIL model 
(loaded from `pretrained` directory) run:  
```python src/predict.py -m pretrained -o scores data/mil/test```

* To evaluate predicted scores and compute statistics (ROC, FPR, TPR, etc.) run:  
```python src/evaluate.py -s scores -o eval_results```

### Retraining MIL model
* Firstly extract features from train videos:  
```python utils/prepare_C3D_features --fast --out_c3d data/c3d/train --out_mil data/mil/train --input_file data/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt```

* Then, train the MIL model on prepared features and save:  
```python src/train.py -s pretrained data/mil/train```
