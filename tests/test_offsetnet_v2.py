import os
import sys
sys.path.insert(0, os.getcwd())

import pytest

import TalkingFace.aligner as aligner

@pytest.mark.net
def test_net():
    import random
    from TalkingFace.aligner import offsetNetV2, torch

    device = "cuda:0"
    landmarks_num = 25
    control_num = 20
    input_tensor = torch.randn((1, 2, 512, 512)).to(device)

    # normal
    net = offsetNetV2(
                      2,
                      control_num,
                      base_channels = 8
                     )
    net.to(device)
    output = net(input_tensor)
    assert output.shape == torch.Size([1, control_num])

