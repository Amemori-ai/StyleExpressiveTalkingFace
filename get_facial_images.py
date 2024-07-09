import os
import cv2
import tqdm
import torch
import click

import numpy as np
from TalkingFace.ExpressiveVideoStyleGanEncoding.\
                 ExpressiveEncoding.train import \
                 validate_video_gen,\
                 StyleSpaceDecoder, \
                 stylegan_path


@click.command()
@click.option('--from_path')
@click.option('--to_path')
def get_facial_images(from_path,
                      to_path):
    """get facial images
    """
    assert os.path.isdir(from_path), "expected directory."

    os.makedirs(to_path, exist_ok = True)
    
    save_video_path = os.path.join(to_path, "video.mp4")

    latent_path = os.path.join(from_path, 'cache.pt')
    latest_decoder_path = None
    facial_attr_path = os.path.join(from_path, "facial")
    face_folder_path = os.path.join(from_path, "data", "smooth")
    gen_file_list, _, _, _ = torch.load(latent_path)
    ss_decoder = StyleSpaceDecoder(stylegan_path = stylegan_path)

    validate_video_gen(
                        save_video_path,
                        latest_decoder_path,
                        facial_attr_path,
                        ss_decoder,
                        len(gen_file_list),
                        face_folder_path
                      )

if __name__ == '__main__':
    get_facial_images()
