import os
import cv2
import click
import tqdm
import re
import pdb

import numpy as np

from TalkingFace.get_disentangle_landmarks import DisentangledLandmarks, landmarks_visualization

@click.command()
@click.option('--from_path')
@click.option('--to_path')
def renorm_landmarks(from_path, 
                     to_path):
    """get landmarks 
    """
    # get landmarks
    assert from_path.endswith("npy"), "from path expected postfix is npy."
    landmarks = np.load(from_path)
    for i in range(5):
        landmarks_to_visual = landmarks[i]
        landmarks_visualization(landmarks_to_visual, os.path.join(os.path.dirname(to_path), f"{i + 1}_to_visualization_origin.png"))

    landmarks_func = DisentangledLandmarks()
    landmarks_renorm = landmarks_func.renorm_landmarks(landmarks)
    # visualization part of landmarks renorm

    for i in range(5):
        landmarks_to_visual = landmarks_renorm[i]
        landmarks_visualization(landmarks_to_visual, os.path.join(os.path.dirname(to_path), f"{i + 1}_to_visualization.png"))

    np.save(to_path, landmarks_renorm)


if __name__ == "__main__":
    renorm_landmarks()
