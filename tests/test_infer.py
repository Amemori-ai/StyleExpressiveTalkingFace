import os
import sys
sys.path.insert(0, os.getcwd())

import pytest


def test_infer():

    config_path = os.path.join(os.getcwd(),"./tests/infer_config.yaml")
    save_path = os.path.join(os.getcwd(),"./tests/infer.mp4")
    from TalkingFace.infer import infer
    infer(config_path, save_path)
