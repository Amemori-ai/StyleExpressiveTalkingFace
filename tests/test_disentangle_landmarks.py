import os
import sys
import pytest
import torch

import cv2
import numpy as np

current_dir = os.getcwd()

@pytest.mark.landmark
def test_landmarks():
    sys.path.insert(0, os.getcwd())
    e4e_latent_path = os.path.join(current_dir, "./tests/id.pt")
    _, selected_id_image, _, _ = torch.load(e4e_latent_path)

    from TalkingFace.get_disentangle_landmarks import DisentangledLandmarks, landmarks_visualization
    get_disentangle_landmarks = DisentangledLandmarks()
    selected_id_image = cv2.resize(selected_id_image, (512,512))
    lm2d = get_disentangle_landmarks(np.uint8(selected_id_image))[0]
    cv2.imwrite(os.path.join(current_dir,"./tests/visualization_ori.png"), selected_id_image) 
    landmarks_visualization(lm2d, os.path.join(current_dir,"./tests/visualization.png"))

@pytest.mark.visual
def test_visualization():
    sys.path.insert(0, os.getcwd())
    lm2d = np.load("/data1/Dataset/chenlong/tx/tx_lm2d_n.npy")
    from TalkingFace.get_disentangle_landmarks import landmarks_visualization
    for idx, _lmd in enumerate(lm2d):
        if idx > 5:
            break
        landmarks_visualization(_lmd, os.path.join(current_dir,f"./tests/visualization_lm2d_{idx+1}.png"))

