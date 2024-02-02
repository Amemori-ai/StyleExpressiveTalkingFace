import os 
import sys
from .aligner import sync_lip_validate, yaml, edict

 
def infer(
            config_path: str,
            save_path: str
         ):

    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    landmarks_path = config.landmarks_path
    net_config = config.net.config
    net_weight = config.net.weight
    
    pti_weight = config.pti.weight
    attribute_weight = config.attr_path
    e4e_path = config.e4e_path

    sync_lip_validate(
                      landmarks_path,
                      net_config,
                      net_weight,
                      pti_weight,
                      attribute_weight,
                      e4e_path,
                      save_path
                     )
