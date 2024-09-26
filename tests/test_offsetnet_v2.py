import os
import sys
sys.path.insert(0, os.getcwd())

import pytest

import TalkingFace.aligner as aligner

@pytest.mark.net
def test_net():
    import random
    from TalkingFace.aligner import offsetNetV2, torch, offsetNetV3

    device = "cuda:0"
    landmarks_num = 25
    control_num = 20
    batchsize = 128
    input_tensor = torch.randn((batchsize, 2, 64, 64)).to(device)

    # normal
    net = offsetNetV3(
                      2,
                      control_num,
                      base_channels = 8,
                      from_size = 64,
                      frames = 4
                     )
    net.to(device)
    output = net(input_tensor)
    assert output.shape == torch.Size([batchsize, control_num])

