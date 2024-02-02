import os
import sys
sys.path.insert(0, os.getcwd())
import torch

def test_landmarks():

    current_dir = os.getcwd()

    e4e_latent_path = os.path.join(current_dir, "./tests/id.pt")
    _, selected_id_image, _, _ = torch.load(e4e_latent_path)



    import cv2
    import numpy as np
    from TalkingFace.get_disentangle_landmarks import DisentangledLandmarks, landmarks_visualization
    get_disentangle_landmarks = DisentangledLandmarks()
    selected_id_image = cv2.resize(selected_id_image, (512,512))
    lm2d = get_disentangle_landmarks(np.uint8(selected_id_image))[0]
    cv2.imwrite(os.path.join(current_dir,"./tests/visualization_ori.png"), selected_id_image) 
    landmarks_visualization(lm2d, os.path.join(current_dir,"./tests/visualization.png"))
