import os
import sys
sys.path.insert(0, os.getcwd())

import cv2
import numpy as np

from functools import reduce
from skimage import transform
from TalkingFace.get_disentangle_landmarks import draw_multiple_landmarks

if __name__ == '__main__':

    landmarks = np.load(sys.argv[1])[:100,:, :2]
    landmark_ref = np.load(sys.argv[2])
    result_path = sys.argv[3]

    os.makedirs(result_path, exist_ok = True)
    n_dim = landmarks.ndim
    if n_dim < 3:
        landmark_to_draw = landmarks[np.newaxis, ...]
    else:
        landmark_to_draw = landmarks

    for i, landmark in enumerate(landmark_to_draw):
        #transform_instance = transform.SimilarityTransform()
        #transform_instance.estimate(landmark[48:68, :], landmark_ref[0][48:68, :])
        #M = transform_instance.params
        #M = np.linalg.inv(transform_instance.params)
        #landmark_copy = landmark.copy()
        #landmark[48:68, :] = np.matmul(landmark[48:68, :], M[:2, :2]) + M[:2, 2]
        
        shift_y = [
                   landmark[54, 1] - landmark_ref[0][54, 1],
                   landmark[60, 1] - landmark_ref[0][60, 1],
                   #landmark[6, 1] - landmark_ref[0][6, 1],
                   #landmark[10, 1] - landmark_ref[0][10, 1]
                  ]
        if reduce(lambda x,y: x + y, shift_y) > 0:
            print(max(*shift_y))
            landmark[:,1] = landmark[:,1] - max(*shift_y)
        else:
            print(min(*shift_y))
            landmark[:,1] = landmark[:,1] - min(*shift_y)


        image = draw_multiple_landmarks([landmark ,landmark_ref[0]])
        cv2.imwrite(os.path.join(result_path, f"output_{i + 1}.jpg"), image)
