import os
import pytest
import yaml
from yaml import Loader


@pytest.fixture(scope='session')
def cfg():
    config_path = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               'config.yaml')
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg


# ignore test_loss_impl.py by default since it requires theano to be installed
collect_ignore = ['test_loss_impl.py']
