import os

import pytest
import torch

DATA = os.path.join(os.path.dirname(__file__), 'data')
SPHERE = os.path.join(DATA, 'sphere.obj')


@pytest.fixture(scope='session')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
