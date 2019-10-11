"""
This script compares C3D features extracted from UCF TEST Dataset
and converted to text files in format expected by MIL model
obtained using scripts from this project and from the original Github repo.
"""
import os
from numpy.testing import assert_array_almost_equal
from src.data_loader import load_features_from_file

py_segmented = 'data/mil/test'
mat_segmented = 'data/mil/testv2_mat'

py_files = sorted(os.listdir(py_segmented))
mat_files = sorted(os.listdir(mat_segmented))

py_files = [os.path.join(py_segmented, f) for f in py_files]
mat_files = [os.path.join(mat_segmented, f) for f in mat_files]

for py_file, mat_file in zip(py_files, mat_files):
    py_feat = load_features_from_file(py_file, n_seg=32)
    mat_feat = load_features_from_file(mat_file, n_seg=32)
    assert_array_almost_equal(py_feat, mat_feat)
