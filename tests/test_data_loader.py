from src.data_loader import normal_abnormal_split


def test_normal_abnormal_split():
    video_list = ['Normal'] * 11 + ['Anomaly'] * 10

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.0)
    assert (len(norm), len(abnorm)) == (11, 0)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.1)
    assert (len(norm), len(abnorm)) == (11, 1)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.2)
    assert (len(norm), len(abnorm)) == (11, 3)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.3)
    assert (len(norm), len(abnorm)) == (11, 5)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.4)
    assert (len(norm), len(abnorm)) == (11, 7)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.5)
    assert (len(norm), len(abnorm)) == (10, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.6)
    assert (len(norm), len(abnorm)) == (7, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.7)
    assert (len(norm), len(abnorm)) == (4, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.8)
    assert (len(norm), len(abnorm)) == (2, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.9)
    assert (len(norm), len(abnorm)) == (1, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=1.0)
    assert (len(norm), len(abnorm)) == (0, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.9999)
    assert (len(norm), len(abnorm)) == (0, 10)

    norm, abnorm = normal_abnormal_split(video_list, abnorm_ratio=0.001)
    assert (len(norm), len(abnorm)) == (11, 0)
