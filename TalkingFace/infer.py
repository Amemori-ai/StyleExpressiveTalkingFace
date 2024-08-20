import os 
import sys
import click
import re
from .aligner import sync_lip_validate, yaml, edict

__all__ = ['infer']
 
def infer(
            config_path: str,
            save_path: str,
            blender: object = None,
            frames: int = -1
         ):

    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    supress = "linear" if not hasattr(config, "supress") else config.supress

    landmarks_path = config.landmarks_path
    landmarks_name = os.path.basename(landmarks_path).split('.')[0].split('_')[-1]

    net_config = config.net.config
    net_weight = config.net.weight
    video_images_path = config.video_images_path
    
    pti_weight = config.pti.weight
    if not pti_weight.endswith('pt') and not pti_weight.endswith("pth"):
        folder = pti_weight
        pti_weight = os.path.join(folder, sorted(os.listdir(pti_weight), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
        print(f"latest weight path is {pti_weight}")

    attribute_weight = config.attr_path
    pose_latent_path = config.pose_latent_path
    id_landmark = config.id_landmark

    net_exp_name = config.net.config.split('/')[-2]
    pti_exp_name = list(filter(lambda x: 'exp' in x or 'pivot' in x or 'pti' in x, pti_weight.split('/')))[0]
    pose_exp_name = pose_latent_path.split('/')[-2]

    if not save_path.endswith("mp4"):
        save_path = os.path.join(save_path, net_exp_name + '_' + pti_exp_name + '_' + pose_exp_name + '_' + landmarks_name + '_' + supress + '.mp4')
    driving_images_dir = None
    if hasattr(config, "driving_images_dir"):
        driving_images_dir = config.driving_images_dir

    sync_lip_validate(
                      landmarks_path,
                      net_config,
                      net_weight,
                      pti_weight,
                      pose_latent_path,
                      attribute_weight,
                      id_landmark,
                      save_path,
                      video_images_path,
                      driving_images_dir,
                      resolution = config.resolution,
                      blender = blender,
                      frames = frames if not hasattr(config, "frames") else config.frames,
                      supress = supress,
                      stylegan_path = None if not hasattr(config, "stylegan_path") else config.stylegan_path
                     )
@click.command()
@click.option('--config_path')
@click.option('--save_path')
def _invoker_infer(
                    config_path,
                    save_path
                  ):
    return infer(config_path, save_path)


if __name__ == '__main__':
    _invoker_infer()

