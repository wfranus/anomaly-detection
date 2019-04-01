import os
import re
import shutil
from numpy.testing import assert_array_almost_equal
from tempfile import mkdtemp, mktemp
from src.data_loader import load_features_from_file
from utils.prepare_C3D_features import (
    create_input_prototxt,
    create_output_prefix_file,
    create_network_prototxt,
    run_C3D_extraction,
    split_c3d_features_to_segments,
    prepare_C3D_features
)


def test_create_input_prototxt():
    input_dir = 'data'
    input_files = ['abnorm/Explosion025_x264.mp4']
    out_dir = mkdtemp()

    create_input_prototxt(input_dir, input_files, out_dir)

    created_file = os.path.join(out_dir, 'input.txt')
    expected_file = os.path.join('tests', 'fixtures', 'input.txt')
    assert os.path.exists(expected_file)
    assert open(created_file, 'r').read() == open(expected_file, 'r').read()

    shutil.rmtree(out_dir)


def test_create_output_prefix_file():
    input_prototxt = os.path.join('tests', 'fixtures', 'input.txt')
    out_dir = mkdtemp()
    out_features_dir = 'out/c3d'

    create_output_prefix_file(input_prototxt, out_dir,
                              out_features_dir=out_features_dir)

    created_file = os.path.join(out_dir, 'output_prefix.txt')
    expected_file = os.path.join('tests', 'fixtures', 'output_prefix.txt')
    assert os.path.exists(expected_file)
    assert open(created_file, 'r').read() == open(expected_file, 'r').read()

    shutil.rmtree(out_dir)


def test_create_network_prototxt(cfg):
    out_dir = mkdtemp()
    out_file = os.path.join(out_dir, 'feature_extraction.prototxt')
    input_file = os.path.join('tests', 'fixtures', 'input.txt')
    batch_size = 32
    mean_file = os.path.join(
        cfg['C3D']['root'],
        'examples',
        'c3d_feature_extraction',
        'sport1m_train16_128_mean.binaryproto'
    )

    create_network_prototxt(out_file, cfg['C3D']['root'], input_file,
                            batch_size=batch_size)

    assert os.path.exists(out_file)
    net_config = open(out_file, 'r').read()
    assert re.search('name: "DeepConv3DNet_Sport1M_Val"', net_config)
    assert re.search(f'source: "{input_file}"', net_config)
    assert re.search(f'mean_file: "{mean_file}"', net_config)
    assert re.search(f'batch_size: {batch_size}', net_config)

    os.remove(out_file)


def test_run_C3D_extraction(cfg):
    network_prototxt_template = 'tests/fixtures/feature_extraction.prototxt'
    network_prototxt = '/tmp/net_config.prototxt'
    input_file = 'tests/fixtures/input.txt'
    output_prefix_file = 'tests/fixtures/output_prefix.txt'
    features_dir = 'out/c3d/Explosion025_x264'
    mean_file = os.path.join(
        cfg['C3D']['root'],
        'examples',
        'c3d_feature_extraction',
        'sport1m_train16_128_mean.binaryproto'
    )

    # prepare network config file
    shutil.copy(network_prototxt_template, network_prototxt)
    with open(network_prototxt, 'r') as f:
        config = f.read()
    config = re.sub(r'(source:) ".*"', r'\1 "' + str(input_file) + '"', config)
    config = re.sub(r'(mean_file:) ".*"', r'\1 "' + str(mean_file) + '"', config)
    config = re.sub(r'(batch_size:) \d+', r'\1 ' + str(cfg['C3D']['batch_size']), config)
    with open(network_prototxt, 'w') as f:
        f.write(config)

    ret_code = run_C3D_extraction(cfg['C3D']['root'], network_prototxt,
                                  cfg['C3D']['model'],
                                  output_prefix_file,
                                  cfg['gpu'],
                                  cfg['C3D']['batch_size'])

    assert ret_code == 0
    assert os.path.isdir(features_dir)
    assert len(os.listdir(features_dir)) == 31

    shutil.rmtree(features_dir)


def test_split_c3d_features_to_segments():
    features_dir = 'data/abnorm/Explosion025_x264'

    num_seg = 32
    out_dir = mkdtemp()

    split_c3d_features_to_segments(features_dir, n_seg=num_seg,
                                   out_dir=out_dir)

    created_file = os.path.join(out_dir, 'Explosion025_x264.txt')
    seg_features = load_features_from_file(created_file, n_seg=num_seg)
    expected_file = 'data/abnorm/Explosion025_x264.txt'
    exp_features = load_features_from_file(expected_file, n_seg=num_seg)
    assert_array_almost_equal(seg_features, exp_features)

    shutil.rmtree(out_dir)


def test_prepare_C3D_features(cfg):
    video_dir = 'data/abnorm'
    input_file = mktemp()
    out_c3d = mkdtemp('c3d')
    out_mil = mkdtemp('mil')
    num_seg = 32

    with open(input_file, 'w') as f:
        f.write('Explosion025_x264.mp4\n')

    prepare_C3D_features(dict(model=cfg['C3D']['model'],
                              video_dir=video_dir,
                              input_file=input_file,
                              out_c3d=out_c3d,
                              out_mil=out_mil,
                              c3d_root=cfg['C3D']['root'],
                              batch_size=cfg['C3D']['batch_size'],
                              num_seg=num_seg,
                              gpu=cfg['gpu'],
                              fast=True))

    assert os.path.isdir(out_mil)
    assert len(os.listdir(out_mil)) == 1
    created_file = os.path.join(out_mil, 'Explosion025_x264.txt')
    assert os.path.isfile(created_file)

    seg_features = load_features_from_file(created_file, n_seg=num_seg)
    assert seg_features.shape == (num_seg, 4096)
    expected_file = os.path.join(video_dir, 'Explosion025_x264.txt')
    exp_features = load_features_from_file(expected_file, n_seg=num_seg)
    assert_array_almost_equal(seg_features, exp_features)

    os.remove(input_file)
    shutil.rmtree(out_c3d)
    shutil.rmtree(out_mil)
