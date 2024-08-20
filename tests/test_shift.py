import os
import sys

import cv2
sys.path.insert(0, os.getcwd())
import numpy as np
import imageio

current_pwd = os.getcwd()


from TalkingFace.aligner import shift_v2, shift_v3

def norm(x):
    scale = 64
    x[..., 0] = x[..., 0] - 151
    x[..., 1] = x[..., 1] - 274

    return ((x / 210 - 0.5) * 2) * scale // 2 + scale // 2

def test_shift():
    landmarks = np.load(os.path.join(current_pwd, "./dataset/exp010/0/train_landmarks.npy"))

    writer = imageio.get_writer(os.path.join(current_pwd, "./tests/shift_v2.mp4"), fps = 25)
    landmarks = np.concatenate((landmarks[:, 6:11,:], landmarks[:, 48:68,:]), axis = 1)
    diff_landmarks = np.zeros_like(landmarks)
    diff_landmarks[:-1] = landmarks[1:] - landmarks[:-1]
    shift = shift_v3(0.2, dim = 'x+y', prob = 0.5, slice = [5, 25])
    landmarks = norm(landmarks)

    for index, landmark in enumerate(landmarks[:, ...]):
        #landmark_shift = shift(landmark.copy())
        landmark_shift = landmark.copy()
        landmark_map = np.zeros((64,64))
        landmark_map_v2 = np.zeros((64,64))
        _landmark_shift = landmark_shift.astype(np.int32)
        _landmark = landmark.astype(np.int32)
        for (point_shift, point) in zip(_landmark_shift[...,:2], _landmark):
            x, y = point_shift
            landmark_map[y, x] = 255
            x, y = point
            landmark_map_v2[y, x] = 255

        writer.append_data(np.concatenate((landmark_map, landmark_map_v2), axis = 0))

        








