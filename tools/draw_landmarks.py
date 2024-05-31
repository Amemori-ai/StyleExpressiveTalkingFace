import os
import sys
sys.path.insert(0, os.getcwd())

import cv2
import numpy as np

from TalkingFace.get_disentangle_landmarks import draw_multiple_landmarks

if __name__ == '__main__':

    landmarks = np.load(sys.argv[1])[:,:, :2]
    landmark_ref = np.load(sys.argv[2])

    n_dim = landmarks.ndim
    if n_dim < 3:
        landmark_to_draw = landmarks[np.newaxis, ...]
    else:
        landmark_to_draw = landmarks

    for i, landmark in enumerate(landmark_to_draw):
        image = draw_multiple_landmarks([landmark, landmark_ref[0]])
        cv2.imwrite(sys.argv[3].replace('.png', f'_{i+1}.png'), image)
