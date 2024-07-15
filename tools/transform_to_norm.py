import os
import sys
sys.path.insert(0, os.getcwd())

import cv2


import numpy as np
from DeepLog import logger

def transform2norm(
                    pred: np.ndarray,
                    key_mean_shape: np.ndarray,
                    stats: np.ndarray,
                    personal_config = dict()
                  ):
    pred[:, :, 1] = 512 - pred[:,:,1]
    preds = ((pred - 256) / 256 - key_mean_shape.reshape((-1, 204)).reshape(-1,68,3)) * 10

    # mean expression, identity, pred_mean, pred_std
    lrs3_idexp_mean = stats['idexp_lm3d_mean'].reshape([1, 68, 3])
    lrs3_idexp_std = stats['idexp_lm3d_std'].reshape([1, 68, 3])
    logger.info(f"pred lm3d : {preds.shape}")
    lm3d_mean = preds.mean(axis=0).reshape([1, 68, 3])
    lm3d_std = preds.std(axis=0).reshape([1, 68, 3])

    # align mouth center (deprecated)
    mouth_center = personal_config.get('mouth_center', None)
    if mouth_center is not None:
        logger.info(f"mask_region.split(',') : {mouth_center.split(',')}")
        cx, cy = [ float(x)  for x in mouth_center.split(',')]
        center = np.array([cx, cy, 0]).reshape((1,1,3))
        mean_ =  preds.reshape((-1,68,3))[:, 48:68, :].mean(axis=1).reshape(-1, 1,3)  # [n, 1, 3]
        shift = mean_ - center
    else:
        shift = np.zeros((1,1,3))

    # mean std shift
    idexp_lm3d = preds.reshape([-1, 68, 3])
    idexp_lm3d -= shift.reshape((-1,1,3))
    idexp_lm3d = (idexp_lm3d  - lm3d_mean) / lm3d_std
    idexp_lm3d= idexp_lm3d * lrs3_idexp_std + lrs3_idexp_mean

    # adjust x_scale (deprecated)
    x_scale = float(personal_config.get("x_scale", 1.0))
    logger.info(f'x_scale : {x_scale}')
    idexp_lm3d = idexp_lm3d.reshape((-1,68,3))
    idexp_lm3d[..., 0] *= x_scale

    # mean shape + exp + id
    idexp_lm3d = idexp_lm3d.reshape([-1,204]) / 10 + key_mean_shape.reshape((-1, 204))
    lm3d = idexp_lm3d.reshape((-1, 68, 3))
    lm3d = lm3d * 256 + 256

    # align to center (deprecated)
    align_center = int(personal_config.get("align_center", 0))
    logger.info(f"align_center : {align_center}")
    if align_center != 0:
        lrs3_512 = lrs3_idexp_mean.reshape([-1,204]) / 10 + self.key_mean_shape.reshape((-1, 204))
        lrs3_512 = lrs3_512.reshape((1,68,3)) * 256 + 256
        lrs3_center = np.mean(lrs3_512[:, 48:68, :], axis=1, keepdims=True)  # (1, 1, 3)
        lm3d_center = np.mean(lm3d[:, 48:68, :], axis=1, keepdims=True) # (n, 1, 3)
        lm3d = lm3d - (lm3d_center - lrs3_center) # (n, 68, 3)
    lm3d[:, :, 1] = 512 - lm3d[:,:,1]
    return lm3d

if __name__ == '__main__':

    key_mean_shape_path = "../efs_model/model/Voice2Lip_v1.3.1/key_mean_shape.npy"
    lrs3_stats_path = "../efs_model/model/Voice2Lip_v1.3.1/lrs3_stats/stats.npy"
    key_mean_shape = np.load(key_mean_shape_path)
    lrs3_stats = np.load(lrs3_stats_path, allow_pickle=True).item()

    preds = np.load(sys.argv[1])
    landmark_ref = np.load(sys.argv[2])

    pred_transformed = transform2norm(preds, key_mean_shape, lrs3_stats)
    landmark_to_draw = pred_transformed[:,:,:2]
    
    #np.save(sys.argv[3], landmark_to_draw)

    preds[:,:, 1] = 512 - preds[:, :, 1]

    from TalkingFace.get_disentangle_landmarks import draw_multiple_landmarks
    for i, landmark in enumerate(landmark_to_draw):
        image = draw_multiple_landmarks([landmark, landmark_ref[0], preds[i][:,:2]])
        cv2.imwrite(sys.argv[3].replace('.png', f'_{i+1}.png'), image)

