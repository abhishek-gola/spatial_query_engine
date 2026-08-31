import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from sqe.data.synthetic import make
from sqe.pipeline import finish_scene


@pytest.fixture(scope="session")
def studio():
    s = make("studio")
    finish_scene(s, verbose=False)
    return s


@pytest.fixture(scope="session")
def square():
    s = make("square")
    finish_scene(s, verbose=False)
    return s
