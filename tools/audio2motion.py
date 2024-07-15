import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

from DeepEngine.jit import JitBase
from DeepLog import logger

from .Timer import TimeRecoder


class Audio2Motion:
    def __init__(self, model_path:str, key_mean_shape_path:str, device='cuda', backend='tensorrt', time_recorder:TimeRecoder=None) -> None:
        self.device = device
        self.backend = backend
        self.model = JitBase(model_path, device = device, backend = self.backend)
        self.key_mean_shape = np.load(key_mean_shape_path)
        self.time_recorder = time_recorder if time_recorder is not None else TimeRecoder()
    
    def get_audio2motion_inputs(self, inp1, inp2, step=1000):
        if inp2.shape[1] % step == 0:
            pad_length = 0
            n = inp2.shape[1] // step
        else:
            pad_length = step - inp2.shape[1] % step
            n = inp2.shape[1] // step + 1

        input_list = []
        in_pad1 = np.pad(inp1, [[0,0],[0,pad_length*2],[0,0]])
        in_pad1[:, inp2.shape[1]*2: ] = inp1[:, -1]
        in_pad2 = np.pad(inp2, [[0,0],[0,pad_length]])
        in_pad2[:, inp2.shape[1]: ] = inp2[:, -1]

        for i in range(n):
            input_list.append((in_pad1[:, i*step*2:(i+1)*step*2], in_pad2[:, i*step:(i+1)*step]))
        return input_list

    def pred_audio2landmarks_norm(self, 
        feat:np.ndarray, 
        mask:np.ndarray, 
        stats:np.ndarray, 
        audio2motion_path:str=None, 
        sigma=0.5, 
        personal_config={}
    ) ->  np.ndarray:
        r"""
        Do the followings:

        1.Pred landmarks by audio features (hubert features).

        2.Shift  mean and std with respect to the stats.

        3.Align if needed. 

        4.Smooth lm3d by gaussian filter if needed.

        Args:
            feat (np.ndarray):  
                Hubert features.
            mask (np.ndarray): 
                Hubert feature masks.
            stats (np.ndarray):  
                Dataset stats (lrs3 + ours). 
            audio2motion_path (str, optional):  
                Model path
            sigma (float, optional): 
                Smooth parameter. Deprecated.
            personal_config (dict, optional): 
                Personal configuration, such as "x_cale", "sigma" ...

        Returns:
            np.ndarray:  3d landmarks with  the shape (n, 68, 3) .
        """
        self.time_recorder.snapshot(f"[Audio2Motion/total]")
        s_len, pred_list = mask.shape[1], []
        inputs = self.get_audio2motion_inputs(feat, mask, step=1000)

        # infer
        for x1, x2 in inputs:
            self.time_recorder.snapshot(f"[Audio2Motion/infer]")
            z_p = np.random.randn(1, 16, 250)
            pred = self.model(x1, x2, z_p, lazy_path=audio2motion_path)
            pred_list.append(pred)
            self.time_recorder.snapshot(f"[Audio2Motion/infer]")
        preds = np.concatenate(pred_list, axis=1)[:, :s_len][0]
        
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
        idexp_lm3d = idexp_lm3d.reshape([-1,204]) / 10 + self.key_mean_shape.reshape((-1, 204))
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

        # smooth landmarks by gaussian filter (deprecated)
        if sigma is not None and sigma > 0:
            logger.info(f"simga:{sigma}")
            lm3d = gaussian_filter1d(lm3d, sigma=sigma, axis=0)

        # opencv coordinates
        preds[:, :, 1] = 512 - preds[:, :, 1]
        self.time_recorder.snapshot(f"[Audio2Motion/total]")
        return lm3d    

    def pred_audio2landmarks(self, 
            feat:np.ndarray,
            mask:np.ndarray,
            audio2motion_path:str=None, 
        ) ->  np.ndarray:
        r"""
        Pred landmarks by audio features (hubert features).

        Args:
            feat (np.ndarray):  
                Hubert features.
            mask (np.ndarray): 
                Hubert feature masks.
            audio2motion_path (str, optional):  
                Model path

        Returns:
            np.ndarray:  3d landmarks with the shape (n, 68, 3) .
        """
        logger.info("[Audio2Motion/pred_audio2landmarks]")
        # infer
        s_len, pred_list = mask.shape[1], []
        inputs = self.get_audio2motion_inputs(feat, mask, step=1000)
        for x1, x2 in inputs:
            self.time_recorder.snapshot("[Audio2Motion/pred]")
            z_p = np.random.randn(1, 16, 250)
            pred = self.model(x1, x2, z_p, lazy_path=audio2motion_path)
            pred_list.append(pred)
            self.time_recorder.snapshot("[Audio2Motion/pred]")
        preds = np.concatenate(pred_list, axis=1)[:, :s_len][0]
        preds = ((preds.reshape((-1, 204)) / 10 + self.key_mean_shape.reshape((-1, 204))).reshape(-1,68,3))  * 256 + 256

        # opencv coordinates
        preds[:, :, 1] = 512 - preds[:, :, 1] 
        return preds