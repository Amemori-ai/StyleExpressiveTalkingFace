import os
import sys
sys.path.insert(0, os.getcwd())

import pytest

@pytest.mark.dataset
def test_dataset():
    current_pwd = os.getcwd()
    config_path = os.path.join(current_pwd,"./tests/config.yaml")
    from TalkingFace.aligner import Dataset, yaml, edict
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    data_config = config.data
    dataset = Dataset(os.path.join(current_pwd, data_config.attr_path), os.path.join(current_pwd, data_config.ldm_path), os.path.join(current_pwd, data_config.id_path), os.path.join(current_pwd, data_config.id_landmark_path))

    print(dataset.get_attr_max())
    #for i in range(len(dataset)):
    #    attr, land = dataset[i]

