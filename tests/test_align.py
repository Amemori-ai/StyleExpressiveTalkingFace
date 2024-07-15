import os
import sys
sys.path.insert(0, os.getcwd())

import pytest

import TalkingFace.aligner as aligner

@pytest.mark.net
def test_net():
    import random
    from TalkingFace.aligner import offsetNet, torch

    device = "cuda:0"
    landmarks_num = 20
    control_num = 20
    input_tensor = torch.randn((1, landmarks_num)).to(device)

    # normal
    net = offsetNet(
                    landmarks_num,
                    control_num
                   )
    net.to(device)
    output = net(input_tensor)
    assert output.shape == torch.Size([1, control_num])

    # depth

    depth = random.choice(list(range(5)))
    net = offsetNet(
                    landmarks_num,
                    control_num,
                    depth = depth
                   )
    net.to(device)
    output = net(input_tensor)
    assert output.shape == torch.Size([1, control_num])

@pytest.mark.dataset
def test_dataset():
    from TalkingFace.aligner import Dataset, yaml, edict
    config_path = "./tests/config.yaml"
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    data_config = config.data
    dataset = Dataset(data_config.attr_path, data_config.ldm_path, data_config.selected_id)

    for i in range(len(dataset)):
        attr, land = dataset[i]

@pytest.mark.aligner
def test_aligner():
    from TalkingFace.aligner import aligner
    config_path = "./tests/config.yaml"
    save_path = "./tests/aligner_test"

    os.makedirs(save_path, exist_ok = True)
    aligner(config_path, save_path)
