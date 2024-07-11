"""get style space latent without mouth and yaw.
"""
import os
import sys
sys.path.insert(0, os.getcwd()) 
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "ExpressiveVideoStyleGanEncoding"))
import torch 
import yaml 
import click 
import tqdm
import numpy as np 
import pickle
from easydict import EasyDict as edict 

from TalkingFace.aligner import update_region_offset, size_of_alpha, alphas
from ExpressiveEncoding.train import StyleSpaceDecoder, stylegan_path

def get_ss_latent(
                   pose_latents_path: str,
                   attributes_path: str,
                   save_path: str
                 ):
    attributes = torch.load(attributes_path)

    decoder = StyleSpaceDecoder(stylegan_path)
    n = len(attributes)
    p_bar = tqdm.tqdm(range(n))
    to_save_list = []

    for i in p_bar:
        attribute = attributes[i]
        w_plus_with_pose = torch.load(os.path.join(pose_latents_path, f'{i + 1}.pt'))
        style_space_latent = decoder.get_style_space(w_plus_with_pose)
        ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to('cuda:0'), [8, len(alphas)])

        to_save_list.append([x.detach().cpu().numpy() for x in ss_updated])

    with open(save_path, 'wb') as f:
        pickle.dump(to_save_list, f, -1)


@click.command()
@click.option('--from_path')
@click.option('--to_path')
def _invoker(from_path,
            to_path):

    with open(from_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    return get_ss_latent(config.pose_path,
                         config.attr_path,
                         to_path
                        )


if __name__ == '__main__':
    _invoker()

