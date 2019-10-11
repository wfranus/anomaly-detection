"""
This script compares predictions on UCF Anomaly Detection TEST Dataset
obtained using implementation of the model from this project and from
the original Github repo
"""
import os
from scipy.io import loadmat
from numpy.testing import assert_array_almost_equal

py_results = 'results'
mat_results = 'results_mat'

py_files = sorted(os.listdir(py_results))
mat_files = sorted(os.listdir(mat_results))

py_files = [os.path.join(py_results, f) for f in py_files]
mat_files = [os.path.join(mat_results, f) for f in mat_files]

for py_file, mat_file in zip(py_files, mat_files):
    print(py_file, mat_file)
    py_feat = loadmat(py_file).get('y_pred')
    mat_feat = loadmat(mat_file).get('y_pred')
    assert py_feat.shape == mat_feat.shape
    assert_array_almost_equal(py_feat, mat_feat, decimal=5)
