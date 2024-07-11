import os
import cv2
import click
import tqdm
import re

import numpy as np


from TalkingFace.get_disentangle_landmarks import DisentangledLandmarks

@click.command()
@click.option('--from_path')
@click.option('--to_path')
@click.option('--shutup', is_flag = True)
def get_landmarks(from_path, 
                  to_path,
                  shutup
                  ):
    """get landmarks 
    """

    to_save_ldm_list = []
    landmarks_func = DisentangledLandmarks()
    if shutup:
        to_save_ldm_list = [landmarks_func(None) for _ in range(250)]
        np.save(to_path, np.concatenate(to_save_ldm_list, axis = 0))
        return 
    
    #assert os.path.isdir(from_path), "expected from_path is directory."
    assert to_path.endswith('npy'), "expected to_path postfix is numpy-like."
    
    # sorted files
    files = os.listdir(from_path)
    files = sorted(files, key = lambda x: int(''.join(re.findall('[0-9]+', x))))
    files = [os.path.join(from_path, x) for x in files]

    p_bar = tqdm.tqdm(files)


    for _file in p_bar:
        image = cv2.imread(_file)
        assert image is not None, f"{_file} not exists."
        ldm = landmarks_func(image)
        to_save_ldm_list.append(ldm)

    np.save(to_path, np.concatenate(to_save_ldm_list, axis = 0))
    

if __name__ == '__main__':
    get_landmarks()
