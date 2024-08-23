"""make a bridge between 
   points and attributes offsets.
"""
import os 
import sys
import cv2
import yaml
import torch
import tqdm
import re
import pdb
import random
import threading
import queue
import time

import torch.nn as nn
import numpy as np

from functools import partial
from typing import Callable, Union, List
from easydict import EasyDict as edict
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from DeepLog import logger, Timer
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from TalkingFace.util import merge_from_two_image, output_copy_region

current_pwd = os.getcwd()

from .get_disentangle_landmarks import DisentangledLandmarks, landmarks_visualization, draw_landmarks, draw_multiple_landmarks
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "ExpressiveVideoStyleGanEncoding"))

from ExpressiveEncoding.train import StyleSpaceDecoder, stylegan_path, from_tensor,\
                                     PoseEdit, get_detector, get_face_info, \
                                     gen_masks, to_tensor, imageio

from .module import BaseLinear, BaseConv2d, Flatten
from .equivalent_offset import fused_offsetNet
from ExpressiveEncoding.loss.FaceParsing.model import BiSeNet

from .extra import exclude

psnr_func = lambda x,y: 20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - y) ** 2)))
psnr_func_np = lambda x,y: 20 * np.log10(255.0 / np.sqrt(np.mean((x - y) ** 2)))
#norm = lambda x: x
#norm = lambda x: np.clip(x / 100, -1, 1)
#norm = lambda x: ((x) / (x.max(axis = (0,1), keepdims = True)))
norm = lambda x: ((x - x.min(axis = (0,1), keepdims = True)) / (x.max(axis = (0,1), keepdims = True) - x.min(axis = (0,1), keepdims = True)) - 0.5) * 2

#norm = lambda x: x / 512
norm_torch = lambda x: ((x - x.amin(axis = (0,1), keepdim = True)) / (x.amax(axis = (0,1), keepdim = True) - x.amin(axis = (0,1), keepdim = True)) - 0.5) * 2

_linear = lambda x: x
_force_linear = lambda x: x
_minmax = lambda x: (((x - x.amin(dim = (0,1), keepdim = True)) / (x.amax(dim = (0,1), keepdim = True) - x.amin(dim = (0,1), keepdim = True))) - 0.5) * 2
_normal = lambda x: (x / 255 - 0.5) * 2
_minmax_inner = lambda x: (((x - x.amin(dim = (1), keepdim = True)) / (x.amax(dim = (1), keepdim = True) - x.amin(dim = (1), keepdim = True) + 1e-4)) - 0.5) * 2

exp_supress = lambda x: (1 / (1 + torch.exp(-4 * (x - 0.5))))
linear_supress = lambda x: x

_minmax_constant_no_clip = lambda x, _min, _max, supress: (supress((x - _min) / (_max - _min)) - 0.5) * 2
_minmax_constant = lambda x, _min, _max: (torch.clip((x - _min) / (_max - _min), 0.0, 1.0) - 0.5) * 2
#_minmax_constant = lambda x, _min, _max: (((x - _min) / (_max - _min)) - 0.5) * 2
_minmax_constant_no_norm = lambda x, _min, _max: torch.clip((x - _min) / (_max - _min), 0.0, 1.0)
_max_constant = lambda x: torch.clip(x / _max, -1.0, 1.0)
_z_constant = lambda x, _mean, _std: (x - _mean) / _std

alpha_indexes = [
                 6, 11, 8, 14, 15, # represent Mouth
                 5, 6, 8, # Chin/Jaw
                 9, 11, 12, 14, 17, # Eyes
                 8, 9 , 11, # Eyebrows
                 9 # Gaze
                ]

alpha_S_indexes = [
                    [113, 202, 214, 259, 378, 501],
                    [6, 41, 78, 86, 313, 361, 365],
                    [17, 387],
                    [12],
                    [45],
                    [50, 505],
                    [131],
                    [390],
                    [63],
                    [257],
                    [82, 414],
                    [239],
                    [28],
                    [6, 28],
                    [30],
                    [320],
                    [409]
                  ]

alphas = [(x, y) for x, y in zip(alpha_indexes, alpha_S_indexes)]
size_of_alpha = 0
for k,v in alphas[:8]:
    size_of_alpha += len(v)

where_am_i = os.path.dirname(os.path.realpath(__file__))
class face_parsing:
    def __init__(self, path = os.path.join(where_am_i, "ExpressiveVideoStyleGanEncoding", "ExpressiveEncoding", "third_party", "models", "79999_iter.pth")):

        net = BiSeNet(19) 
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)
        net.eval()
        net.to("cuda:0")
        self.net = net
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):

        h, w = x.shape[:2]
        x = Image.fromarray(x)
        image = x.resize((512, 512), Image.BILINEAR)
        img = self.to_tensor(image).unsqueeze(0).to("cuda:0")
        out = self.net(img)[0].detach().squeeze(0).cpu().numpy().argmax(0)
        mask = np.zeros_like(out).astype(np.float32)
        for label in list(range(1,  7)) + list(range(10, 14)):
            mask[out == label] = 1
        out = cv2.resize(mask, (w,h))
        return out

class augmentation:
    def __init__(self, 
                 prob = 0.2
                ):
        self.prob = prob

    def _identity(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if random.random() <= self.prob:
            return self.forward(*args, **kwargs)
        return self._identity(*args, **kwargs) 

class shift(augmentation):
    def __init__(
                 self, 
                 pixel_range : list = [-20, 20],
                 dim : str = 'all',
                 prob = 0.2
                ):
        super().__init__(prob)
        self.pixel_range = list(range(pixel_range[0], pixel_range[1] + 1))
        self.dim = dim

    def __repr__(self):
        return f'shift pixel_range {self.pixel_range} dim {self.dim}'

    def _get_cord(self, x):
        return x + random.choice(self.pixel_range)

    def _identity(self, cords):
        return cords

    def forward(self, cords):

        dim = self.dim
        if dim == 'x':
            cords[:, 0] = self._get_cord(cords[:,0])
            return cords
        if dim == 'y':
            cords[:, 1] = self._get_cord(cords[:,1])
            return cords
        if dim == 'x+y' or dim == 'all':
            cords[:, 0] = self._get_cord(cords[:, 0])
            cords[:, 1] = self._get_cord(cords[:, 1])
            return cords

        raise RuntimeError(f"dim {dim} type not exits.")

class shift_v2(shift):
    def __init__(
                  self,
                  scale: float = 1, # 2 pixel shift
                  dim : str = 'all',
                  diff: List = None,
                  prob : float = 0.2
                ):
        super().__init__(dim = dim, prob = prob)
        self.scale = scale
        self.diff = diff

    def __repr__(self):
        return f'\
                  scale: {self.scale} \
                '
    def _identity(self, cords, *args):
        return cords

    def _get_cord(
                  self, 
                  x: np.ndarray,
                  index: int
                 ):
        return x + (np.random.random(x.shape)) * self.scale * self.diff[index, ...]

    def forward(self, cords, index):
        return self._get_cord(cords, index)

class shift_v3(shift):
    def __init__(
                  self,
                  scale: float = 1, # 2 pixel shift
                  dim : str = 'all',
                  prob : float = 0.2,
                  slice : List = None
                ):
        super().__init__(dim = dim, prob = prob)
        self.scale = scale
        self.slice = slice

    def __repr__(self):
        return f'\n\
                  scale: {self.scale}\n \
                  slice: {self.slice}\n \
                '
    def _identity(self, cords, *args):
        return cords

    def _get_cord(
                  self, 
                  x: np.ndarray,
                 ):
        if self.slice is not None:
            f, e = self.slice
            x[f:e, :] = x[f:e, :] + (np.round(np.random.random(x[f:e, :].shape), decimals = 7) - 0.5) * 2 * self.scale
        else:
            x = x + (np.round(np.random.random(x.shape), decimals = 7) - 0.5) * 2 * self.scale
        return x

    def forward(self, cords):
        return self._get_cord(cords)

class shift_v4(shift):
    def __init__(
                  self,
                  scale: float = 1, # 2 pixel shift
                  dim : str = 'all',
                  landmark: List = None,
                  prob : float = 0.2
                ):
        super().__init__(dim = dim, prob = prob)
        self.scale = scale
        self.landmark = landmark

    def __repr__(self):
        return f'\
                  scale: {self.scale} \
                '
    def _identity(self, cords, *args):
        return cords

    def _get_cord(
                  self, 
                  x: np.ndarray,
                  shift: np.ndarray
                 ):
        return x + (np.random.random(x.shape) - 0.5) * 2 * self.scale * shift

    def forward(self, cords, shift):
        return self._get_cord(cords, shift)


class shrinkage(augmentation):
    pass

class expansion(augmentation):
    pass

class affine(augmentation):

    # ops

    # shift

    # shear

    # rotate

    def __init__(
                 self, 
                 affine = [dict(_type = "scale", kwargs = dict(_range = [0.25, 0.5, 2, 4], isotropy = True))],
                 prob = 0.2
                ):
        super().__init__(prob)
        self.affine = affine

    def __repr__(self):
        return f'affine {self.affine}'
    
    def _scale(self, x, _range = [1, 1],  isotropy = True):
        if isotropy:
            scale = random.choice(_range)
        else:
            scale = np.array([random.choice(_range) for _ in range(2)]).reshape(1, 2)
        return x * scale

    def _identity(self, cords):
        return cords
        
    def forward(self, cords):

        #affine_matrix = np.zeors((3,3)).astype(np.float32)

        for affine_config in self.affine:
            cords = getattr(self, "_" + affine_config["_type"])(cords, **affine_config["kwargs"])
        return cords

class SmoothL1LossMyself(torch.nn.SmoothL1Loss):
    def forward(self, x, y):
        #x = torch.nn.functional.softmax(x)
        #y = torch.nn.functional.softmax(y)
        return super().forward(x,y)

class offsetNet(nn.Module):
    def __init__(
                 self, 
                 in_channels,
                 out_channels,
                 depth = 0,
                 base_channels = 512,
                 **kwargs
                ):
        super().__init__()

        dropout = kwargs.get("dropout", False)
        skip = kwargs.get("skip", False)
        norm = kwargs.get("norm", "BatchNorm1d")

        renorm = kwargs.get("renorm", "clip")
        remap = kwargs.get("remap", False) 
        use_fourier = kwargs.get("use_fourier", False)
        self.use_fourier = use_fourier

        norm_type = '_' + kwargs.get("norm_type", 'linear')
        norm_func = eval(norm_type)

        if norm_type == '_minmax_constant' \
           or norm_type == "_minmax_constant_no_norm" \
           or norm_type == "_minmax_constant_no_clip" \
           or norm_type == "_force_linear":
            norm_dim = kwargs.get("norm_dim",  [0, 1])
            if norm_dim == [0, 1]:
                dim_size = 1
            else:
                dim_size = in_channels // 2

            if '_max' in kwargs and '_min' in kwargs:
                _min, _max = torch.tensor(kwargs['_min'], dtype = torch.float32), torch.tensor(kwargs['_max'], dtype = torch.float32)
            else:
                _min = torch.zeros((1,dim_size,2))
                _max = torch.zeros((1,dim_size,2))
            self.register_buffer('_min', _min)
            self.register_buffer('_max', _max)
        if use_fourier:
            if "pos" in kwargs:
                pos = kwargs["pos"][None, :, :]
            else:
                pos = np.zeros((1, in_channels // 2, 2))
            self.register_buffer("pos", torch.tensor((pos / 512).astype(np.float32)))
            in_channels *= 2

        self._build_net(in_channels, base_channels, out_channels, depth, **kwargs)

        self.norm_type = norm_type
        self.norm = norm_func

        self.register_buffer("clip_values", torch.ones((out_channels, 2), dtype = torch.float32))
        if renorm == "repa":
            def repa(x):
                self.clip_values.to(x)
                return (torch.clip(x, -1.0, 1.0) * 0.5 + 0.5) * (self.clip_values[:, 1] - self.clip_values[:,0]) + self.clip_values[:, 0]
            self.renorm = repa
        elif renorm == "clip":
            def clip(x):
                self.clip_values.to(x)
                return torch.clip(x, self.clip_values[:,0], self.clip_values[:, 1])
            self.renorm = clip
        elif renorm == "znorm":
            self.register_buffer("z_values", torch.ones((2, out_channels), dtype = torch.float32))
            def renorm(x):
                self.z_values.to(x)
                return torch.clip(x * self.z_values[1, :] + self.z_values[0, :],  self.clip_values[:,0], self.clip_values[:, 1])
                #return x * self.z_values[1, :] + self.z_values[0, :]
            self.renorm = renorm
        else:
            raise RuntimeError("unexpected renorm function.")

    def _build_net(self,
                   in_channels,
                   base_channels,
                   out_channels,
                   depth,
                   skip = False,
                   norm = "BatchNorm1d",
                   dropout = False,
                   remap = False,
                   **kwargs
                  ):
        if depth > 0:
            _modules = [nn.Linear(in_channels, base_channels) if not kwargs.get("first_convbn", False) else BaseLinear(in_channels, base_channels, skip = skip, norm = norm)] + ([torch.nn.Dropout1d(0.5)] if dropout else []) + \
                       [*([BaseLinear(base_channels, base_channels, skip = skip, norm = norm)] +  ([torch.nn.Dropout1d(0.5)] if dropout else []))] * depth + \
                       [*([nn.Linear(base_channels, out_channels)] + ([torch.nn.Dropout1d(0.5)] if dropout else []))]
        else:
            _modules = [nn.Linear(in_channels, base_channels), nn.Linear(base_channels, out_channels)] #nn.ReLU(),
        if remap:
            _modules += [nn.Linear(out_channels, out_channels, bias = False)]
        self.net = nn.Sequential(*_modules)

    def set_clip(self, clip_values: torch.Tensor):
        self.clip_values = clip_values

    def set_z_attr(self, values: torch.Tensor):
        self.z_values = values

    def forward(self, x):
        
        if self.norm_type == '_minmax_constant' or self.norm_type == "_minmax_constant_no_norm" or self.norm_type == "_minmax_constant_no_clip":
            if self.norm_type == "_minmax_constant_no_clip":
                x = self.norm(x, self._min, self._max, exp_supress)
            else:
                x = self.norm(x, self._min, self._max)
        else:
            x = self.norm(x)

        n = x.shape[0]
        if hasattr(self, "pos"):
            x = torch.cat((x, self.pos.repeat(n,1 ,1)), dim = 1)

        x = x.reshape(n, -1)
        y = self.net(x)
        return torch.clip(y * self.z_values[1, ...] + self.z_values[0, ...], self.clip_values[0, ...], self.clip_values[1, ...])
        #return self.renorm(y)

class offsetNetV2(offsetNet):

    def _build_net(
                   self,
                   in_channels,
                   base_channels,
                   out_channels,
                   depth,
                   skip = False,
                   norm = "BatchNorm2d",
                   dropout = False,
                   remap = False,
                   **kwargs
                  ):

        LOG2 = lambda x: np.log10(x) / np.log10(2)

        from_size = kwargs.get("from_size", 128)
        target_size = kwargs.get("target_size", 1)
        max_channels = kwargs.get("max_channels", 128)
        act = kwargs.get("act", "ReLU")

        self.from_size = from_size

        layers = int(LOG2(from_size)) - int(LOG2(target_size))

        channels = [base_channels * (2 ** i) for i in range(layers)]
        channels = [x if x <= max_channels else max_channels for x in channels]

        class BlurModule(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.module_3x3 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups = in_channels // 2, bias = False)
                self.module_5x5 = nn.Conv2d(in_channels // 2, in_channels // 2, 5, 1, 2, groups = in_channels // 2, bias = False)
                self.module_7x7 = nn.Conv2d(in_channels // 2, in_channels // 2, 7, 1, 3, groups = in_channels // 2, bias = False)

                self.merge = nn.Conv2d(in_channels // 2 * 4 + 1 , in_channels, 1, 1, 0)
            def forward(self, x):
                y = x[:,0:1, ...]
                o_x3 = self.module_3x3(y)
                o_x5 = self.module_5x5(y)
                o_x7 = self.module_7x7(y)
                return self.merge(torch.cat((y, o_x3, o_x5, o_x7, x[:, 1:2, ...]), dim = 1))

        _modules = [
                    #BlurModule(in_channels),
                    nn.Conv2d(in_channels,  base_channels, 3, 1, 1, bias = False) if not kwargs.get("first_convbn", False) else BaseConv2d(in_channels, base_channels, 3, 1, 1, act = None, bias = False)
                   ] 

        in_c = base_channels
        for i in range(layers):
            out_c = channels[i]
            _modules += [BaseConv2d(in_c, out_c, 3, 2, 1, act = act, norm = norm, repeat = depth)]
            in_c = out_c
        _modules += [Flatten()]
        if remap:
            _modules += [nn.Linear(in_c * (target_size ** 2), in_c), nn.Linear(in_c, in_c), nn.Linear(in_c, out_channels)]
        else:
            _modules += [nn.Linear(in_c, out_channels)]
        if dropout: 
            _modules += [torch.nn.Dropout1d(0.1)]
        self.net = nn.Sequential(*_modules)

    def forward(self, x):
    
        if self.norm_type == '_minmax_constant' or self.norm_type == "_minmax_constant_no_norm" or self.norm_type == "_minmax_constant_no_clip":
            if self.norm_type == "_minmax_constant_no_clip":
                x = self.norm(x, self._min, self._max, exp_supress)
            else:
                x = self.norm(x, self._min, self._max)
        else:
            x = self.norm(x)
        y = self.net(x)
        return torch.clip(y * self.z_values[1, ...] + self.z_values[0, ...], self.clip_values[..., 0], self.clip_values[..., 1])
    #return self.renorm(y)

class Dataset:
    def __init__(
                 self,
                 attributes_path: str, # final file is a pt-format file.
                 ldm_path: str, # final file is a numpy-format file.
                 id_path: str,
                 id_landmark_path :str, 
                 augmentation: bool = False,
                 norm_dim: list = [0, 1],
                 is_abs: bool = False,
                 is_flow_map: bool = False,
                 use_point: bool = False
                ):

        assert os.path.exists(attributes_path), f"attribute path {attributes_path} not exist."
        assert os.path.exists(ldm_path), f"attribute path {ldm_path} not exist."

        gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(id_path)

        self.attributes = torch.load(attributes_path, map_location = "cpu")
        # convert to cpu and requires_grad False.
        for i, attr in enumerate(self.attributes):
            if isinstance(attr, list):
                x, y = attr
                x.requires_grad = False
                x = x.to('cpu')
                for idx, _y in enumerate(y):
                    temp = _y.detach()
                    temp.requires_grad = False
                    temp = temp.to('cpu')
                    y[idx] = temp
                self.attributes[i] = [x, y]
            else:
                attr.requires_grad = False
        landmarks = np.load(ldm_path)
        #assert len(attributes) == len(landmarks), "attributes length unequal with landmarks."
        id_landmark = np.load(id_landmark_path)
        self.augmentation = augmentation
        if augmentation:
            ops = []
            self.ops = ops
            logger.info([op for op in self.ops])
        self.is_flow_map = is_flow_map
        if is_flow_map:
            #landmarks = (landmarks - landmarks.mean(axis = (0, 1), keepdims = True)) / (landmarks.std(axis = (0,1), keepdims = True) + 1e-8)
            self.offset_range = self.get_offset_range(np.concatenate(((landmarks)[:, 6:11, :], (landmarks)[:, 48:68, :]), axis = 1), norm_dim, is_abs)
            class get_maps:
                def __init__(
                              self, 
                              _type = "polylines", 
                              min_max: List = None
                            ):

                    assert _type in ["polylines", "points"]
                    self.type = _type
                    self.scale = 64
                    self.pad = 0
                    self.min, self.max = min_max
                    _landmarks = np.concatenate((landmarks[:, 6:11,:], landmarks[:, 48:68,:]), axis = 1)
                    diff_landmarks = np.zeros_like(_landmarks)
                    landmarks_renorm = self.renorm(_landmarks, dim = 1)
                    diff_landmarks[:-1] = landmarks_renorm[1:] - landmarks_renorm[:-1]
                    #self.op = shift_v3(scale = 2.0, dim = 'x+y', prob = 1.0, diff = diff_landmarks)
                    self.op = shift_v3(scale = 1.0, dim = 'x+y', prob = 0.5)

                def renorm(self, x, dim = 0):
                    scale = self.scale
                    pad = self.pad
                    return 2 * ((x - x.min(axis = dim, keepdims = True)) / (x.max(axis = dim, keepdims = True) - x.min(axis = dim, keepdims = True)) - 0.5) * scale / 2 + scale / 2 + pad / 2

                def renorm_v2(self, x):
                    _min, _max = self.min[0], self.max[0]
                    scale = self.scale
                    pad = self.pad
                    return 2 * (np.clip((x - _min) / (_max - _min), 0.0, 1.0) - 0.5) * scale / 2 + scale / 2 + pad / 2

                def renorm_v3(self, x):
                    scale = 64
                    x[..., 0] = x[..., 0] - 151
                    x[..., 1] = x[..., 1] - 274

                    return ((x / 210 - 0.5) * 2) * scale // 2 + scale // 2

                def __getitem__(self, index):

                    _id_landmarks = self.renorm_v3(np.concatenate((id_landmark[0, 6:11,:],id_landmark[0, 48:68,:]), axis = 0))
                    _landmarks = self.renorm_v3(np.concatenate((landmarks[index, 6:11,:],landmarks[index, 48:68,:]), axis = 0))
                    
                    if augmentation:
                        _landmarks = self.op(_landmarks)

                    scale = self.scale
                    pad = self.pad

                    id_map = np.zeros((scale + pad, scale + pad, 3), dtype = np.uint8) #(np.random.random((scale + pad, scale + pad, 3)) * 200).astype(np.uint8)
                    landmark_map = np.zeros((scale + pad, scale + pad, 3), dtype = np.uint8)#(np.random.random((scale + pad , scale + pad, 3)) * 200).astype(np.uint8)

                    if self.type == "polylines":
                        cv2.polylines(id_map, [_id_landmarks[0:5,:].astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.polylines(id_map, [_id_landmarks[5:25,:].astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.polylines(landmark_map, [_landmarks[0:5,:].astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.polylines(landmark_map, [_landmarks[5:25,:].astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
                        #landmark_map = cv2.GaussianBlur(landmark_map, (5,5), 0.5)
                    elif self.type == "points":
                        _id_landmarks = _id_landmarks.astype(np.int32)
                        _landmarks = _landmarks.astype(np.int32)
                        for (point_i, point) in zip(_id_landmarks, _landmarks):
                            x_i, y_i = point_i
                            x, y = point
                            id_map[y_i, x_i, :] = 255
                            landmark_map[y, x, :] = 255
                    return ((np.concatenate((landmark_map[..., 0:1], id_map[..., 0:1]), axis = 2) / 255.0 - .5) * 2).transpose((2, 0, 1)) 
            self.offsets = get_maps(_type = "points" if use_point else "polylines", min_max = self.offset_range)
            """
            class get_maps:
                def __getitem__(self, index):
                    id_map = np.zeros((512, 512, 3), np.uint8)
                    cv2.polylines(id_map, [id_landmark[0, 6:11,:].astype(np.int32)], False, (255, 255, 255), 2)
                    cv2.polylines(id_map, [id_landmark[0, 48:68,:].astype(np.int32)], True, (255, 255, 255), 2)

                    landmark_map = np.zeros((512, 512, 3), np.uint8)
                    cv2.polylines(landmark_map, [landmarks[index, 6:11,:].astype(np.int32)], False, (255, 255, 255), 2)
                    cv2.polylines(landmark_map, [landmarks[index, 48:68,:].astype(np.int32)], True, (255, 255, 255), 2)
                    return ((np.concatenate((landmark_map[..., 0:1], id_map[..., 0:1]), axis = 2) / 255.0 - .5) * 2).transpose((2, 0, 1)) 
            """
        else:
            self.offsets = np.concatenate(((landmarks - id_landmark)[:, 48:68, :], (landmarks - id_landmark)[:, 6:11, :]), axis = 1)
            self.offset_range = self.get_offset_range(self.offsets, norm_dim, is_abs)

        #self.offsets = norm(self.offsets)
        self.length = len(self.attributes) #round(len(self.attributes) * ratio)
        self.landmarks = landmarks
        current_folder = "/data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/"
        self.gen_files = [os.path.join(current_folder, x) for x in gen_file_list]

    def __len__(self):
        return self.length

    def get_attr_z(self):

        mean, std = None, None

        eps = 1e-6

        if isinstance(self.attributes[0][1], list):

            _array = np.array([x[1][:size_of_alpha] for x in self.attributes])
            mean = _array.mean(axis = 0, keepdims = True)
            std = _array.std(axis = 0, keepdims = True) + eps
        else:

            mean = self.attributes[:, 1, :size_of_alpha].mean(axis = 0, keepdims = True)
            std = self.attributes[:, 1, :size_of_alpha].std(axis = 0, keepdims = True)

        return torch.tensor(np.concatenate((mean, std), axis = 0))

    def get_attr_max(self):
        if isinstance(self.attributes[0][1], list):
            clip_values = [[0xffff,0] for _ in range(len(self.attributes[0][1][:size_of_alpha]))]
            for attr in self.attributes:
                attr = attr[1]
                for i in range(len(attr[:size_of_alpha])):
                    clip_values[i][0] = min(clip_values[i][0], attr[i])
                    clip_values[i][1] = max(clip_values[i][1], attr[i])
            clip_values = np.array(clip_values)
        else:
            clip_values = np.zeros_like(self.attributes[0][1])[:,np.newaxis].repeat((1, 2))
            for i, attr in enumerate(self.attributes):
                attr = attr[1][:size_of_alpha]
                clip_values[:, 0] = np.minimum(clip_values[:, 0], attr)
                clip_values[:, 1] = np.maximum(clip_values[:, 1], attr)
        return torch.tensor(clip_values)

    def get_offset_range(self, x, norm_dim, is_abs = False):
        if is_abs:
            x = np.abs(x)
        return np.min(x, axis = tuple(norm_dim), keepdims = True), np.max(x, axis = tuple(norm_dim), keepdims = True)

    def __getitem__(
                    self,
                    index
                   ):
    
        if isinstance(self.attributes[index], list):
            attributes = [] 
            for x in self.attributes[index][1][:size_of_alpha]:
                x.requires_grad = False
                attributes.append(x)
            attribute = torch.stack(attributes, dim = 0)
        else:
            attribute = self.attributes[index][:size_of_alpha]
        
        if self.augmentation and not self.is_flow_map:
            for op in self.ops:
                self.offsets[index] = op(self.offsets[index])

        return attribute, \
               torch.from_numpy(self.offsets[index]).to(torch.float32) 

class ValDataset:
    def __init__(self,

                 dataset: object
                ):
        self.dataset = dataset
        self.length = len(dataset.attributes) - len(dataset)
        self.sampler = list(range(len(dataset), len(dataset) + self.length))

    def __len__(self):
        return self.length

    def __getitem__(self,
                       idx: int
                   ):    
    
        idx = self.sampler[idx]
        return self.dataset[idx]

def update_lip_region_offset(
                              dlatents,
                              offset,
                              version = "v1"
                           ):
    if version == 'v1':
        return update_region_offset(dlatents, offset, [0,8])
    if version == 'v2':
        return update_region_offset_v2(dlatents, offset, [0,8])

def update_region_offset(
                          dlatents,
                          offset,
                          region_range
                        ):
    dlatents_tmp = [latent.clone() for latent in dlatents]
    count = 0
    #first 5 elements.
    #forbidden_list = [
    #                  ( 6, 378 ),
    #                  ( 5, 50 ),
    #                  ( 5, 505 )
    #                 ]
    #forbidden_list = []
    #if tuple([k, i]) in forbidden_list:
    #    logger.info(f"{k} {i} is forbidden.")
    #    continue
    for k, v in alphas[region_range[0]:region_range[1]]:
        for i in v:
            dlatents_tmp[k][:, i] = dlatents[k][:, i] + offset[:,count]
            count += 1
    return dlatents_tmp

def update_region_offset_v2(
                          dlatents,
                          offset,
                          region_range
                        ):
    dlatents_tmp = [torch.zeros_like(latent) for latent in dlatents]
    count = 0
    forbidden_list = []
    #forbidden_list = [
    #                  ( 6, 378 ),
    #                  ( 5, 50 ),
    #                  ( 5, 505 )
    #                 ]
    #first 5 elements.
    for k, v in alphas[region_range[0]:region_range[1]]:
        for i in v:
            if tuple([k, i]) in forbidden_list:
                logger.info(f"{k} {i} is forbidden.")
                continue
            dlatents_tmp[k][:, i] = offset[:,count]
            count += 1
    return [(x, y) for (x,y) in zip(dlatents, dlatents_tmp)]

def aligner(
            config_path: str,
            save_path: str,
            resume_path: str = None
           ):

    import time
    from tqdm import tqdm
    from tensorboardX import SummaryWriter
    from torchvision.utils import make_grid

    with open(os.path.join(current_pwd, config_path), encoding = 'utf-8') as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    net_config = config.net
    name = "offsetNet" if not hasattr(net_config, "name") else net_config.name
    use_point = False if not hasattr(net_config, "use_point") else net_config.use_point

    snapshots = os.path.join(save_path, "snapshots")
    tensorboard_path = os.path.join(save_path, "tensorboard", f"{time.time()}")
    os.makedirs(snapshots, exist_ok = True)
    os.makedirs(tensorboard_path, exist_ok = True)
    device = "cuda:0"
    writer = SummaryWriter(tensorboard_path)
    decoder = StyleSpaceDecoder(stylegan_path).to(device)
    if hasattr(config, "stylegan_path"):
        logger.info(config.stylegan_path)
        decoder.load_state_dict(torch.load(config.stylegan_path))
    pti_path = None if not hasattr(config, "pti_path") else config.pti_path
    if pti_path is not None:
        decoder.load_state_dict(torch.load(pti_path))
        logger.info(f" decoder loaded from {pti_path} ")

    for p in decoder.parameters():
        p.requires_grad = False

    datasets_list = []
    renorm = "clip" if not hasattr(config.net, "renorm") else config.net.renorm
    _reload = True if not hasattr(config, "reload") else config.reload

    for _config in config.data:
        dataset_config = _config.dataset
        dataset = Dataset(os.path.join(current_pwd, dataset_config.attr_path),\
                          os.path.join(current_pwd,dataset_config.ldm_path),\
                          os.path.join(current_pwd, dataset_config.id_path), \
                          os.path.join(current_pwd, dataset_config.id_landmark_path), \
                          augmentation = False if not hasattr(dataset_config,"augmentation") else dataset_config.augmentation, \
                          norm_dim = [0, 1] if not hasattr(config.net, "norm_dim") else config.net.norm_dim,
                          is_abs = (config.net.norm_type == "minmax_constant_no_norm"),
                          is_flow_map = False if not hasattr(dataset_config,"is_flow_map") else dataset_config.is_flow_map,
                          use_point = use_point
                        )
        datasets_list.append(dataset)
    
        clip_values = dataset.get_attr_max()
        if renorm == "znorm":
            values = dataset.get_attr_z()
        offset_range = dataset.offset_range
    dataset = torch.utils.data.ConcatDataset(datasets_list)
    dataloader = DataLoader(dataset, batch_size = config.batchsize, shuffle = True, num_workers = 8, drop_last = True)
    training_batchsize = config.batchsize
    dataset_config = config.val
    val_dataset = Dataset(
                           os.path.join(current_pwd, dataset_config.attr_path), 
                           os.path.join(current_pwd,dataset_config.ldm_path), 
                           os.path.join(current_pwd, dataset_config.id_path), 
                           os.path.join(current_pwd, dataset_config.id_landmark_path),
                           is_flow_map = False if not hasattr(dataset_config,"is_flow_map") else dataset_config.is_flow_map,
                           use_point = use_point,
                           augmentation = False if not hasattr(dataset_config,"augmentation") else dataset_config.augmentation
                         )
    val_dataloader = DataLoader(val_dataset, batch_size = config.batchsize, shuffle = False)

    # init net
    is_refine = False if not hasattr(net_config, "is_refine") else net_config.is_refine
    remap = False if not hasattr(net_config, "remap") else net_config.remap
    use_fourier = False if not hasattr(net_config, "use_fourier") else net_config.use_fourier
    pos = None
    if use_fourier:
        pos = np.load(os.path.join(current_pwd, dataset_config.id_landmark_path))
        pos = np.concatenate(((pos)[0, 48:68, :], (pos)[0, 6:11, :]), axis = 0)


    net = eval(name)(\
                    net_config.in_channels * 2 if name == "offsetNet" else net_config.in_channels, size_of_alpha, \
                    net_config.depth, \
                    base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , 
                    max_channels = 128 if not hasattr(net_config, "max_channels") else net_config.max_channels , \
                    dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                    norm = "BatchNorm1d" if not hasattr(net_config, "norm") else net_config.norm, \
                    skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                    is_refine = is_refine, \
                    norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                    _min = offset_range[0], \
                    _max = offset_range[1],
                    norm_dim = [0, 1] if not hasattr(net_config, "norm_dim") else net_config.norm_dim,
                    renorm = renorm,
                    remap = remap,
                    use_fourier = use_fourier,
                    pos = pos,
                    from_size = 512 if not hasattr(net_config, "from_size") else net_config.from_size,
                    act = "ReLU" if not hasattr(net_config, "act") else net_config.act,
                    first_convbn = False if not hasattr(net_config, "first_convbn") else net_config.first_convbn,
                    target_size = 1 if not hasattr(net_config, "target_size") else net_config.target_size
                   )
    logger.info(net)
    best_nme = 100.0
    best_acc = 0.0
    if resume_path is not None:
        logger.info(f"resume training from {resume_path}")
        state_dict = torch.load(resume_path)
        weight = state_dict
        if "best_value" in state_dict :
            weight = state_dict['weight']
            if _reload:
                best_nme = state_dict['best_value']
            logger.info(f"load weight and nme : {best_nme}")
        net.load_state_dict(weight, False)
    net.train()
    net.set_clip(clip_values)
    if renorm == "znorm":
        net.set_z_attr(values)
    net.to(device)
    # enable calculate derivate.
    for p in net.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(net.parameters(), lr = net_config.lr) #Adam(net.parameters(), lr = net_config.lr)
    sche_name = "StepLR" if not hasattr(config.optim,"sche") else config.optim.sche
    sche = getattr(torch.optim.lr_scheduler, sche_name)
    if sche_name == "StepLR":
        sche = sche(optimizer, config.optim.step, gamma = config.optim.gamma)
    else:
        sche = sche(optimizer, base_lr = config.optim.base_lr, max_lr = config.optim.max_lr, step_size_up = config.optim.step, cycle_momentum=False)
    loss_d = torch.nn.CosineSimilarity(dim = 1)
    loss_i = torch.nn.L1Loss() #SmoothL1LossMyself() #torch.nn.MSELoss()

    start_epoch = 1 if not hasattr(config, "start_epoch") else config.start_epoch
    pbar = tqdm(range(start_epoch, config.epochs + start_epoch))
    total_count = 0
    last_path = None


    #weights = []
    #for (k, i) in alphas[0:8]:
    #    for _i in i:
    #        weight = decoder.get_conv_weight(k, _i).reshape(1, 1, -1) 
    #        weight = (weight.square().sum(dim = [2]) + 1e-8).rsqrt().reshape(1,1)
    #        weights.append(weight)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in pbar:
        for idx, data in enumerate(dataloader):
            with torch.autocast(device_type = "cuda", dtype = torch.float16):
                optimizer.zero_grad()
                attr, offset = data
                attr = attr.to(device)
                n = attr.shape[0]
                offset = offset.to(device)
                d_loss_value = 0.0
                pred_attr = net(offset)
                #weight_loss_value = 0
                #for weight in weights:
                #    weight = weight.repeat(n, 1, 1)
                #    weight_loss_value += loss_i(weight * attr.reshape(n, -1, 1), weight * pred_attr.reshape(n, -1, 1))
                attr = ((attr - net.clip_values[:, 0]) / (net.clip_values[:, 1] - net.clip_values[:, 0]))
                pred_attr = (pred_attr - net.clip_values[:,0]) / (net.clip_values[:,1] - net.clip_values[:,0])

                d_loss_value = 1 - loss_d(pred_attr, attr).mean() 
                i_loss_value = loss_i(attr, pred_attr)

                weight_loss_value = 0
                loss_value = i_loss_value * 1.0 + d_loss_value * 0.0 + weight_loss_value * 1.0
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            total_count += 1

            if total_count % config.show_internal == 0:
                #logger.info(f"epoch:{epoch}: {idx+1}/{len(dataloader)} loss {loss_value.mean().item()}, d_loss {d_loss_value.mean().item()} i_loss {i_loss_value.mean().item()} weight_loss {weight_loss_value.mean().item()} ")
                logger.info(f"epoch:{epoch}: {idx+1}/{len(dataloader)} loss {loss_value.mean().item()}, d_loss {d_loss_value.mean().item()} i_loss {i_loss_value.mean().item()} ")
                writer.add_scalar("loss", loss_value.mean().item(), total_count)
                writer.add_image(f'image', make_grid(offset[:, 0:1, ...].detach(),normalize=True, scale_each=True), epoch)

        nme_value = 0.0
        acc_value = 0.0
        sim_value = 0.0
        # validate 
        net.eval()
        for idy, data in enumerate(val_dataloader):
            attr, offset = data
            attr = attr.to(device)
            offset = offset.to(device)
            with torch.no_grad():
                pred_attr = net(offset)

            if isinstance(pred_attr, tuple):
                pred_attr = pred_attr[0]

            acc_value += (torch.sign(pred_attr) == torch.sign(attr)).to(torch.float32).mean()
            attr = (attr - net.clip_values[:, 0]) / (net.clip_values[:, 1] - net.clip_values[:, 0])
            pred_attr = (pred_attr - net.clip_values[:,0]) / (net.clip_values[:,1] - net.clip_values[:,0])

            nme_value += torch.nn.functional.mse_loss(attr, pred_attr).mean()
            sim_value += torch.nn.functional.cosine_similarity(attr, pred_attr).mean()

        nme_value /= len(val_dataloader)
        acc_value /= len(val_dataloader)
        sim_value /= len(val_dataloader)
        rnme_value = torch.sqrt(nme_value).item()
        nme_value = nme_value.item()
        acc_value = acc_value.item()
        sim_value = sim_value.item()
        logger.info(f"nme is {nme_value}, acc is {acc_value}, sim is {sim_value} rnme is {rnme_value}.")
        writer.add_scalars("metric", \
                            dict( \
                                 nme = nme_value, \
                                 acc = acc_value, \
                                 sim = sim_value, \
                                 rnme = rnme_value \
                                ), \
                            global_step = epoch)
        net.train()
        if nme_value < best_nme:
            last_path =  os.path.join(snapshots, "best.pth")
            best_nme = nme_value
            torch.save(dict(weight = net.state_dict(), best_value = best_nme, epoch = epoch), last_path)
            logger.info(f"{epoch} weight saved.")
        sche.step()
        writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'] ,global_step = epoch)
    return last_path

def get_gen_image(
                    id_ss_latent: torch.tensor,
                    offset: list, 
                    decoder: Callable 
                 ):
    ss_latent = id_ss_latent
    if offset is not None:
        ss_latent = update_lip_region_offset(id_ss_latent, offset)
    return decoder(ss_latent)

def sync_lip_validate(
                      landmarks: Union[str , np.ndarray],
                      config_path: str,
                      offset_weight_path: str,
                      pti_weight_path: str,
                      pose_latent_path: str,
                      attributes_path: str,
                      id_landmark_path: str,
                      save_path: str,
                      video_images_path: str,
                      driving_images_dir: str = None,
                      **kwargs
                     ):

    supress = kwargs.get("supress", "linear")
    _stylegan_path = kwargs.get("stylegan_path", None)
    assert save_path.endswith('mp4'), "expected postfix mp4, but got is {save_path}."

    device = "cuda:0"

    exp_name = os.path.dirname(config_path).split('/')[-1]
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    net_config = config.net
    renorm = "clip" if not hasattr(net_config, "renorm") else net_config.renorm
    remap = False if not hasattr(net_config, "remap") else net_config.remap
    use_fourier = False if not hasattr(net_config, "use_fourier") else net_config.use_fourier
    use_point = False if not hasattr(net_config, "use_point") else net_config.use_point

    name = "offsetNet" if not hasattr(net_config, "name") else net_config.name
    net = eval(name)(\
                    net_config.in_channels * 2 if name == "offsetNet" else net_config.in_channels, size_of_alpha, \
                    net_config.depth, \
                    base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , \
                    max_channels = 128 if not hasattr(net_config, "max_channels") else net_config.max_channels , \
                    dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                    norm = "BatchNorm1d" if not hasattr(net_config, "norm") else net_config.norm, \
                    skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                    norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                    norm_dim = [0, 1] if not hasattr(net_config, "norm_dim") else net_config.norm_dim, \
                    renorm = renorm ,\
                    remap = remap,
                    use_fourier = use_fourier,
                    from_size = 512 if not hasattr(net_config, "from_size") else net_config.from_size,
                    act = "ReLU" if not hasattr(net_config, "act") else net_config.act,
                    first_convbn = False if not hasattr(net_config, "first_convbn") else net_config.first_convbn,
                    target_size = 1 if not hasattr(net_config, "target_size") else net_config.target_size
                    )
    net.to(device)
    state_dict = torch.load(offset_weight_path)
    weight = state_dict
    if "best_value" in state_dict :
        weight = state_dict['weight']
        best_nme = state_dict['best_value']
        logger.info(f"load weight and nme : {best_nme}")
    net.load_state_dict(weight, False)
    net.eval()
    resolution = kwargs.get("resolution", 1024)
    decoder = StyleSpaceDecoder(stylegan_path, to_resolution = resolution)
    #if _stylegan_path is not None:
    #    logger.info(_stylegan_path)
    #    decoder.load_state_dict(torch.load(_stylegan_path), False)
    #from collections import OrderedDict
    #new_dict = OrderedDict()
    #for k,v in torch.load(pti_weight_path).items():
    #    if "affine" not in k:
    #        new_dict[k] = v
    decoder.load_state_dict(torch.load(pti_weight_path), False)
    decoder.to(device)
    attributes = torch.load(attributes_path, map_location="cpu")

    #get_disentangled_landmarks = DisentangledLandmarks()
    #id_path = e4e_latent_path.replace("pt", "txt")
    #if os.path.exists(id_path):
    #    with open(id_path, 'r') as f:
    #        selected_id = int(f.readlines()[0].strip())
    #else:
    #    _, selected_id_image, selected_id_latent, selected_id = torch.load(e4e_latent_path)
    #    with open(id_path, 'w') as f:
    #        f.write(str(selected_id))

    if isinstance(landmarks, str):
        landmarks = np.load(landmarks)[...,:2]
    #landmarks[..., 1] = 512 - landmarks[..., 1]
    landmarks = gaussian_filter1d(landmarks, sigma=1.0, axis=0)
    # hard code id video landmarks.
    id_landmarks = np.load(os.path.join(current_pwd, config.val.id_landmark_path))

    shift_y = [
               landmarks[:, 54, 1] - id_landmarks[:, 54, 1],
               landmarks[:, 60, 1] - id_landmarks[:, 60, 1],
               #landmarks[:, 6, 1] - id_landmarks[:, 6, 1],
               #landmarks[:, 10, 1] - id_landmarks[:, 10, 1]
              ]
    
    shift_y = np.stack(shift_y, axis = 1)
    shift_empty = np.zeros((landmarks.shape[0], 1))
    plus = shift_y.mean(axis = 1) > 0
    shift_empty[plus, 0] = shift_y[plus, :].max(axis = 1)
    minus = shift_y.mean(axis = 1) < 0
    shift_empty[minus, 0] = shift_y[minus, :].min(axis = 1)
    shift_y = shift_empty

    #landmarks[:,:,1] = landmarks[:,:,1] - shift_y.reshape(-1, 1)
    landmarks = np.concatenate((landmarks[:, 6:11,:],landmarks[:, 48:68,:]), axis = 1)
    id_landmarks = np.concatenate((id_landmarks[:, 6:11,:],id_landmarks[:, 48:68,:]), axis = 1)
    #landmarks = (landmarks - landmarks.mean(axis = (0, 1), keepdims = True)) / (landmarks.std(axis = (0,1), keepdims = True) + 1e-8)
    if net_config.in_channels == 25:
        landmark_offsets = landmarks - id_landmarks
    else:
        landmark_offsets = (landmarks - id_landmarks)[:, 5:25, :]

    driving_files = None
    if driving_images_dir is not None:
        assert os.path.isdir(driving_images_dir), "expected directory."
        driving_files = sorted(os.listdir(driving_images_dir), key = lambda x: int(''.join(re.findall('[0-9]+', x))))
        driving_files = [os.path.join(driving_images_dir, x) for x in driving_files]

    assert os.path.isdir(video_images_path), "expected directory."
    image_files = sorted(os.listdir(video_images_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))
    image_files = [os.path.join(video_images_path, x) for x in image_files]

    frames = kwargs.get("frames", -1)
    n = min(landmark_offsets.shape[0], len(attributes), len(image_files), len(driving_files)) if frames == -1 else frames
    logger.info(f"length is {n}..")
    t = Timer()

    batch_size = 8
    bundle_datasets = [i for i in range(0, n, batch_size)]

    p_bar = tqdm.tqdm(bundle_datasets)
    detector = get_detector()
    face_parse_func = face_parsing()

    class get_pose:
        def __init__(
                     self,
                     pose_latent_path
                    ):
            if os.path.isdir(pose_latent_path):
                self.pose = pose_latent_path
            else:
                self.pose = torch.load(pose_latent_path, map_location = "cpu")

        def __call__(self, i):
            if isinstance(self.pose, str):
                return torch.load(os.path.join(pose_latent_path, f'{i + 1}.pt'), map_location = "cpu")
            else:
                return self.pose[i]

    class get_maps:
        
        def __init__(
                     self, 
                     _type = "polylines",
                     min_max: List = None
                    ):

            assert _type in ["polylines", "points"]
            self.type = _type
            self.min, self.max = min_max

        def renorm(self, x, scale, pad):
            return 2 * ((x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0)) - 0.5) * scale / 2 + scale / 2 + pad / 2

        def renorm_v2(self, x, scale, pad):
            _min, _max = self.min[0], self.max[0]
            return 2 * (np.clip((x - _min) / (_max - _min), 0.0, 1.0) - 0.5) * scale / 2 + scale / 2 + pad / 2

        def renorm_v3(self, x, scale, pad):
            x_copy = x.copy()
            x_copy[..., 0] = x_copy[..., 0] - 151
            x_copy[..., 1] = x_copy[..., 1] - 274

            return ((x_copy / 210 - 0.5) * 2) * scale // 2 + scale // 2

        def __getitem__(self, index):

            scale = 64
            pad = 0
            _id_landmarks = self.renorm_v3(id_landmarks[0], scale, pad)
            _landmarks = self.renorm_v3(landmarks[index], scale, pad)
    
            id_map = np.zeros((scale + pad, scale + pad, 3), dtype = np.uint8) #(np.random.random((scale + pad, scale + pad, 3)) * 200).astype(np.uint8)
            landmark_map = np.zeros((scale + pad , scale + pad, 3), dtype = np.uint8)

            canvas = np.zeros((scale + pad, scale + pad, 3), dtype = np.uint8)

            if self.type == "polylines":
                cv2.polylines(id_map, [_id_landmarks[0:5,:].astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.polylines(id_map, [_id_landmarks[5:25,:].astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.polylines(landmark_map, [_landmarks[0:5,:].astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.polylines(landmark_map, [_landmarks[5:25,:].astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.polylines(canvas, [_id_landmarks[0:5,:].astype(np.int32)], False, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.polylines(canvas, [_id_landmarks[5:25,:].astype(np.int32)], True, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.polylines(canvas, [_landmarks[0:5,:].astype(np.int32)], False, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.polylines(canvas, [_landmarks[5:25,:].astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
                

                target_path = os.path.join(current_pwd, "debug", f"{index + 1}.jpg")
                os.makedirs(os.path.dirname(target_path), exist_ok = True)
                cv2.imwrite(target_path, canvas)

                #landmark_map = cv2.GaussianBlur(landmark_map, (5,5), 0.5)
            elif self.type == "points":
                _id_landmarks = _id_landmarks.astype(np.int32)
                _landmarks = _landmarks.astype(np.int32)
                for (point_i, point) in zip(_id_landmarks, _landmarks):
                    x_i, y_i = point_i
                    x, y = point
                    id_map[y_i, x_i, :] = 255
                    landmark_map[y, x, :] = 255
            return ((np.concatenate((landmark_map[..., 0:1], id_map[..., 0:1]), axis = 2) / 255.0 - .5) * 2).transpose((2, 0, 1)) 
     
    if name == "offsetNetV2":
        landmark_offsets = get_maps(_type = "points" if use_point else "polylines", min_max = [net._min.cpu().numpy(), net._max.cpu().numpy()])
    get_pose_func = get_pose(pose_latent_path)
    psnr_value = 0.0
    psnr_values = []

    
    with imageio.get_writer(save_path, fps = 25) as writer:
        batch_size = 8

        w_plus_with_pose = get_pose_func(0).detach() #torch.load(os.path.join(pose_latent_path, f'{i + 1}.pt'))
        style_space_latent = decoder.get_style_space(w_plus_with_pose.to(device))
        q = queue.Queue()
        q2 = queue.Queue()

        psnr_values = []
        psnr_value = 0.0
        output_queue = []

        lock = threading.Lock()

        def kernel():
            j = q.get()
            attribute = attributes[j]
            w_plus_with_pose = get_pose_func(j).detach().to(device) #torch.load(os.path.join(pose_latent_path, f'{i + 1}.pt'))
            style_space_latent = decoder.get_style_space(w_plus_with_pose)
            landmark_offset_np = landmark_offsets[j]
            landmark_offset[j % batch_size] = torch.from_numpy(landmark_offset_np).unsqueeze(0).float().to(device) 
            region_ss = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to(device), [8, len(alphas)])
            #region_ss = update_region_offset(style_space_latent, torch.tensor(attribute[1]).reshape(1, -1).to(device), [0, len(alphas)])
            for index, _  in enumerate(ss_updated):
                ss_updated[index][j % batch_size, ...] = region_ss[index]

        def kernel2():
                
            j = q2.get()
            output = from_tensor(outputs[ j  % batch_size ])
            h,w = output.shape[:2]
            if driving_files is not None:
                landmark_offset_np = landmark_offsets[j]
                image = cv2.imread(image_files[j])[...,::-1]
                w = h = 512
                image = cv2.resize(image, (w,h))
                output = cv2.resize(output, (w,h))
                mask = face_parse_func(image)
                blender = kwargs.get("blender", None)
                output = merge_from_two_image(image, output, mask = mask, blender = blender)
                
                ref_image = cv2.imread(driving_files[j])[..., ::-1]
                ref_image = cv2.resize(ref_image, (w,h))

                if j in exclude:
                    merge_mask = np.zeros_like(output)
                    for (y1,y2, x1, x2) in output_copy_region:
                        merge_mask[y1:y2, x1:x2, ...] = 1.0
                    value = psnr_func_np(image * merge_mask, output * merge_mask)
                    #psnr_value = psnr_value +  value
                    psnr_values.append(value)

                if name != "offsetNetV2":
                    landmark = landmarks[j, ...]
                    image_landmark = draw_multiple_landmarks([landmark ,id_landmarks[0]])
                else:
                    image_landmark = cv2.resize((landmark_offset_np[0:1, :, :].repeat(3, axis = 0).transpose((1,2,0)) * 0.5 + 0.5) * 255, (w,h))
                output = np.concatenate((output, ref_image, image_landmark), axis = 0)
            output_queue[j % batch_size] = output
        
        for i in p_bar:
            end = min(i + batch_size, n)
            size = end - i
            landmark_offset = torch.zeros((size, 2, 64, 64)).to(device)
            ss_updated = [torch.zeros_like(x).repeat(size, 1).to(device) for x in style_space_latent]
            [q.put(j) for j in range(i, end)]
            output_queue = [None for _ in range(size)]
            workers = [threading.Thread(target = kernel, args = ()) for j in range(size)]
            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()
            #for j in range(i, end):
            #if i not in exclude:
            #    continue
            
            with torch.no_grad():
                offset = net(landmark_offset)
            ss_updated = update_lip_region_offset(ss_updated, offset)
            
            """
            pti_training_folder = os.path.join(current_pwd, "pti")
            os.makedirs(pti_training_folder, exist_ok = True)
            [torch.save([x.detach().cpu() for x in ss_updated], os.path.join(pti_training_folder, f"{jj + 1}.pt")) for jj in range(i, end)]
            """
            outputs = torch.clip((decoder(ss_updated) * 0.5 + 0.5) * 255.0, 0.0, 255.0)
            [q2.put(j) for j in range(i, end)]
            workers_out = [threading.Thread(target = kernel2, args = ()) for j in range(size)]
            for worker in workers_out:
                worker.start()
            for worker in workers_out:
                worker.join()
            for index, output in enumerate(output_queue):
                #j = index + i
                #if j in exclude:
                #    target_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0], "video")
                #    os.makedirs(target_path, exist_ok = True)
                #    cv2.imwrite(os.path.join(target_path, f"{exclude.index(j) + 1}.jpg"), np.uint8(output)[..., ::-1])
                writer.append_data(np.uint8(output))
        """
        psnr_list = np.array(psnr_values)
        n = len(exclude)
        with open("psnr.txt", 'a+') as f:
            f.write(f"[{exp_name}] psnr : {psnr_list.mean()}, min: {psnr_list.min()}, max: {psnr_list.max()} \n")

        blank = " " * 8
        with open("psnr_total.txt", "r+") as f:
            lines = f.readlines()
            if len(lines):
                f.seek(0, 0)
                line = lines[0].strip() + blank + f"{exp_name}\n"  
                f.write(line)
                blank = " " * 4
                for line, value in zip(lines[1:], psnr_values):
                    line = line.strip()
                    line = line + blank + str(round(value, 3)) + "\n"
                    f.write(line)
            else:
                f.write(f"pic_id" + blank + f"{exp_name}\n")
                blank = " " * 4
                for index, value in zip(exclude, psnr_values):
                    line = f"{index}" + blank + str(round(value, 3)) + "\n"
                    f.write(line)
        logger.info(f"psnr : {psnr_list.mean()}, min: {psnr_list.min()}, max: {psnr_list.max()}")
        """

def evaluate(
             config_path: str,
             offset_weight_path: str,
             pti_weight_path: str
            ):

    """evaluate data.
    """
    
    device = "cuda:0"
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    net_config = config.net

    name = "offsetNet" if not hasattr(net_config, "name") else net_config.name
    remap = False if not hasattr(net_config, "remap") else net_config.remap
    use_fourier = False if not hasattr(net_config, "use_fourier") else net_config.use_fourier
    net = eval(name)(\
                    net_config.in_channels * 2 if name == "offsetNet" else net_config.in_channels, size_of_alpha, \
                    net_config.depth, \
                    base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , \
                    max_channels = 128 if not hasattr(net_config, "max_channels") else net_config.max_channels , \
                    dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                    norm = "BatchNorm1d" if not hasattr(net_config, "norm") else net_config.norm, \
                    skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                    norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                    norm_dim = [0, 1] if not hasattr(net_config, "norm_dim") else net_config.norm_dim,
                    remap = remap,
                    use_fourier = use_fourier,
                    from_size = 512 if not hasattr(net_config, "from_size") else net_config.from_size,
                    act = "ReLU" if not hasattr(net_config, "act") else net_config.act,
                    first_convbn = False if not hasattr(net_config, "fist_convbn") else net_config.first_convbn,
                    target_size = 1 if not hasattr(net_config, "target_size") else net_config.target_size
                   )

    net.to(device)
    state_dict = torch.load(offset_weight_path)
    weight = state_dict
    if "best_value" in state_dict :
        weight = state_dict['weight']
        best_nme = state_dict['best_value']
        logger.info(f"load weight and nme : {best_nme}")
    net.load_state_dict(weight, False)
    net.eval()

    folder = os.path.dirname(pti_weight_path)
    pti_weight = os.path.join(folder, sorted(os.listdir(pti_weight_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
    decoder = StyleSpaceDecoder(stylegan_path, to_resolution = 512)
    decoder.load_state_dict(torch.load(pti_weight), False)
    decoder.to(device)

    pose_path = config.pose_path
    attr_path = config.attr_path
    poses = torch.load(os.path.join(current_pwd, pose_path))
    attributes = torch.load(os.path.join(current_pwd, attr_path)) 
    attribute = attributes[0]
    
    dataset_config = config.val
    val_dataset = Dataset(
                          os.path.join(current_pwd, dataset_config.attr_path), 
                          os.path.join(current_pwd,dataset_config.ldm_path), 
                          os.path.join(current_pwd, dataset_config.id_path), 
                          os.path.join(current_pwd, dataset_config.id_landmark_path),
                          is_flow_map = False if not hasattr(dataset_config,"is_flow_map") else dataset_config.is_flow_map
                         )
    val_dataloader = DataLoader(val_dataset, batch_size = 8, shuffle = False)
    pbar = tqdm.tqdm(val_dataloader)
    psnr_value = 0.0
    psnr_list = []
    for idy, data in enumerate(pbar):
        attr, offset = data
        attr = attr.to(device)
        offset = offset.to(device)
        with torch.no_grad():
            pred_attr = net(offset)

        if isinstance(pred_attr, tuple):
            pred_attr = pred_attr[0]
        with torch.no_grad():
            n = attr.shape[0]
            w_with_pose = poses[idy].repeat(n,1,1)
            style_space_latent = decoder.get_style_space(w_with_pose)
            ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to(device).repeat(n, 1), [8, len(alphas)])
            ss_updated_pred = update_lip_region_offset(ss_updated, pred_attr, version = 'v2')
            ss_updated_gt = update_lip_region_offset(ss_updated, attr, version = 'v2')
            image = decoder(ss_updated_pred)
            image_gt = decoder(ss_updated_gt)
        cur_psnr = psnr_func(image, image_gt)
        psnr_list += [x.mean().item() for x in cur_psnr[:]]
        psnr_value += cur_psnr.mean().item()#psnr_func(image, image_gt).mean().item()

    psnr_value /= len(val_dataloader)
    psnr_list = np.array(psnr_list)
    logger.info(psnr_value)
    logger.info(psnr_list)
    
    return psnr_list


