import os
import sys
import subprocess
import shutil
import cv2
import re
import PIL
import imageio
import mediapipe as mp
import os.path as osp
import scipy
import numpy as np
import face_alignment
import random 
import time 
import subprocess
import zipfile

where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, where_am_i)
os.chdir(where_am_i)

import deep_3drecon

from scipy.ndimage import gaussian_filter1d
from glob import glob
from DeepLog import logger
from pathlib import Path
from error import *
from GlintCloud.error import CloudError, G_ERROR_SYSTEM_OTHER

from typing import Callable, Union

def draw_landmarks(landmarks):
    """draw landmarks function
    """
    eye_idx = list(range(6, 11))
    mouth_idx = list(range(48, 68))
    pts1, pts2 = [], []
    canvas = np.zeros((512, 512, 3), np.uint8)
    for i in range(68):
        x, y = landmarks[i]
        if i in eye_idx:
            pts1.append((x, y - 30))
            #landmarks[i][1] = y - 30
        elif i in mouth_idx:
            pts2.append((x, y + 10))
            #landmarks[i][1] = y + 10
    pts1 = np.array(pts1, np.int32)
    pts2 = np.array(pts2, np.int32)
    cv2.polylines(canvas, [pts1], False, (255, 255, 255), 2)
    cv2.polylines(canvas, [pts2], True, (255, 255, 255), 2)
    return canvas

def draw_multiple_landmarks(landmarks):
    """draw landmarks function
    """
    eye_idx = list(range(6, 11))
    mouth_idx = list(range(48, 68))

    colors = [
            
                (255,255,255),
                (255,0,0),
                (0, 255, 0),
                (0,0, 255)
             ]

    canvas = np.zeros((512, 512, 3), np.uint8)
    for idx, landmark in enumerate(landmarks):
        pts1, pts2 = [], []
        for i in range(68):
            x, y = landmark[i]

            if i in eye_idx:
                pts1.append((x, y - 30))
                #landmarks[i][1] = y - 30
            elif i in mouth_idx:
                pts2.append((x, y))
                #landmarks[i][1] = y + 10

                #cv2.circle(canvas, (x, y), 2, (255,255,255), -1)
            if i in [6, 10, 60, 54]:
                cv2.putText(canvas, f'{i}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        pts1 = np.array(pts1, np.int32)
        pts2 = np.array(pts2, np.int32)
        color = colors[idx % len(colors)]
        cv2.polylines(canvas, [pts1], False, color, 2)
        cv2.polylines(canvas, [pts2], True, color, 2)
    return canvas


def landmarks_visualization(
                             landmarks: list,
                             save_path: str
                           ):
    """landmark visualization.
    """
    canvas = draw_landmarks(landmarks)
    cv2.imwrite(save_path, canvas)


class DisentangledLandmarks:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.fa = face_alignment.FaceAlignment(1, network_size=4, device='cuda')
        self.face_reconstructor = deep_3drecon.Reconstructor()

        self.key_mean_shape = np.load("data_npy/key_mean_shape.npy").reshape(1, -1)
        self.id_base = np.load("data_npy/key_id_base.npy")
        self.exp_base = np.load("data_npy/key_exp_base.npy")
        self.lrs3_stats = np.load("data_npy/lrs3_stats.npy", allow_pickle=True).item()
        self.lrs3_idexp_mean = self.lrs3_stats['idexp_lm3d_mean'].reshape([1, 68, 3])
        self.lrs3_idexp_std = self.lrs3_stats['idexp_lm3d_std'].reshape([1, 68, 3])
        self.steps = 20
        self.temp_steps = 10
        self.total_time = None
        self.train_video_save_dir = None
        self.temp_video_save_dir = None
        self.dataset_dir = None
        self.temp_5_dir = None
        self.th_list = [
            0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85,
            86, 87,
            88, 89, 90, 91,
            95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304,
            306,
            307, 308, 310, 311, 312,
            314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415]

        self.th_list2 = [57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393, 164, 167, 165, 92, 186]
        self.left_eye = [33, 7, 163, 144, 145, 472, 153, 154, 155, 133, 173, 157, 158, 470, 159, 160, 161, 246, 471,
                         468,
                         469]
        self.left_eye2 = [130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30, 247]
        self.left_eye3 = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]
        self.right_eye = [382, 381, 380, 477, 374, 373, 390, 249, 263, 466, 388, 387, 386, 475, 385, 384, 398, 362, 473,
                          474,
                          476]
        self.right_eye2 = [463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414, ]
        self.right_eye3 = [464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413]

    def lm68_2_lm5(self, in_lm):
        # in_lm: shape=[68,2]
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        # 将上述特殊角点的数据取出，得到5个新的角点数据，拼接起来。
        lm = np.stack([in_lm[lm_idx[0], :], np.mean(in_lm[lm_idx[[1, 2]], :], 0), np.mean(in_lm[lm_idx[[3, 4]], :], 0),
                       in_lm[lm_idx[5], :], in_lm[lm_idx[6], :]], axis=0)
        # 将第一个角点放在了第三个位置
        lm = lm[[1, 2, 0, 3, 4], :2]
        return lm

    def get_3dmm_coeff(self, 
                       frame_bgr: np.ndarray
                      ):


        lm68_lst = []
        lm5_lst = []
        cnt = 0

        coeff_lst = []
        params = []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            lm68 = self.fa.get_landmarks(frame_rgb)[0]  # 识别图片中的人脸，获得角点, shape=[68,2]
        except Exception as e:
            logger.error("Setp 7 检测人脸失败!")
            raise FaceNoDetectedError

        lm5 = self.lm68_2_lm5(lm68)
        lm68_lst.append(lm68)
        lm5_lst.append(lm5)
        coeff, align_img, param = self.face_reconstructor.recon_coeff_and_trans(frame_rgb[None, ...],
                                                                                    lm5[None, ...],
                                                                                    return_image=True)
        coeff_lst.append(coeff)
        params.append(param)

        coeff_arr = np.concatenate(coeff_lst, axis=0)
        params = np.concatenate(params, axis=0)
        return coeff_arr

    def reconstruct_idexp_lm3d(self, id_coeff, exp_coeff, add_mean_face=False):

        identity_diff_face = id_coeff @ self.id_base.transpose(1, 0)  # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = exp_coeff @ self.exp_base.transpose(1, 0)  # [t,c],[c,3*68] ==> [t,3*68]

        face = identity_diff_face + expression_diff_face  # [t,3N]
        face = face.reshape([face.shape[0], -1, 3])  # [t,N,3]
        if add_mean_face:
            lm3d = face + self.key_mean_shape.reshape([1, 68, 3])  # [3*68, 1] ==> [1, 3*68]
        else:
            lm3d = face * 10
        return lm3d

    def lm2d_to_lm3d_like(self, lm2d, face_size = 512):
        lm2d /= float(face_size)
        lm2d[:, :, 1] = 1 - lm2d[:, :, 1]
        lm2d = lm2d * 2 - 1
        return lm2d

    def denorm_lm3d_to_lm2d(self, lm3d, face_size=512):
        """
            lm2d = (lm2d + 1.0) / 2.0 \n
            lm2d[:, :, 1] = 1.0 - lm2d[:, :, 1]
        """
        lm2d = lm3d[:, :, :2]
        lm2d = (lm2d + 1.0) / 2.0
        lm2d[:, :, 1] = 1.0 - lm2d[:, :, 1]
        lm2d *= face_size
        return lm2d

    def coeff2lm2d(self, coeff, use_norm=False):
        if coeff is None:
            return self.denorm_lm3d_to_lm2d(self.key_mean_shape.reshape([1, 68, 3]))

        if use_norm:
            lm3d_std = lm3d_diff.std(axis=0).reshape([1, 68, 3])

            # norm & denorm
            lm3d_diff = (lm3d_diff - lm3d_mean) / lm3d_std
            lm3d_diff = lm3d_diff * self.lrs3_idexp_std + self.lrs3_idexp_mean

            # diff -> lm3d/lm3d
            lm3d = self.key_mean_shape.reshape(1, 68, 3) + lm3d_diff.reshape(-1, 68, 3) / 10
            lm2d = self.denorm_lm3d_to_lm2d(lm3d)
        else:
            lm3d = self.reconstruct_idexp_lm3d(coeff[:, :80], coeff[:, 80:144], add_mean_face=True)
            lm2d = self.denorm_lm3d_to_lm2d(lm3d)

        # center alignment
        #lrs3_face = (self.lrs3_idexp_mean / 10.0 + self.key_mean_shape.reshape((1, 68, 3)))
        #lrs3_face = self.denorm_lm3d_to_lm2d(lrs3_face)
        #lrs3_center = np.mean(lrs3_face[:, 48:68, :2], axis=1, keepdims=True)  # (1,1, 3)
        #lm2d_center = np.mean(lm2d[:, 48:68, :], axis=1, keepdims=True)  # (1,1, 3)
        #lm2d = lm2d - (lm2d_center - lrs3_center)  # (n,68, 3)
        return lm2d

    def renorm_landmarks(self, landmarks):
        if landmarks.dtype != np.float32:
            landmarks = np.float32(landmarks)

        landmarks = self.lm2d_to_lm3d_like(landmarks)
        _diff = (landmarks - self.key_mean_shape.reshape((1,68,3))[:,:,:2]) * 10
        _diff = (_diff - _diff.mean(axis = 0).reshape(1,68,2)) / _diff.std(axis = 0).reshape(1,68,2)
        _diff_to_denorm = _diff * self.lrs3_idexp_std[:,:,:2] + self.lrs3_idexp_mean[:,:,:2]
        landmarks_renorm = _diff_to_denorm / 10 + self.key_mean_shape.reshape(1, 68, 3)[:,:,:2]
        return self.denorm_lm3d_to_lm2d(landmarks_renorm)

    def get_landmark(
                     self,
                     face: Union[str , np.ndarray]
                    ):
    
        self.temp_steps = 0
        
        if isinstance(face, str):
            face = cv2.imread(face)
            assert face is not None, "face path not exits"
        if face is None:
            return self.coeff2lm2d(None)
        # Setp 7 Get 3dmm coeff file
        coef = self.get_3dmm_coeff(face)
        return self.coeff2lm2d(coef)
    

    def __call__(
            self,
            face: Union[str , np.ndarray]
        ):
        return self.get_landmark(
                                 face
                                )

