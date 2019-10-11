# anomaly-detection
Anomaly detection using deep multiple instance learning (MIL)

## Requirements
* Python 3.6+
* C3D v1.0
* Keras 2.2.4 with Tensorflow 1.9.0 backend

## Installation
Follow instructions in `INSTALL.md` to install C3D and its dependencies.
To install Python dependencies run:  
```pip install -r requirements.txt```

To verify that installation was successful run tests:  
```pytest tests```

## Running
To extract C3D features and convert them to format accepted by the MIL model
adjust `config.yml` and run:  
```python utils/prepare_C3D_features --fast --out_c3d data/c3d --out_mil data/mil```

To train the MIL model on prepared features and save it to file run:  
```python src/train.py -s pretrained data```

To predict anomaly scores for test videos using pretrained MIL model run:
```python src/predict.py ```
