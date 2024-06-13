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

import torch.nn as nn
import numpy as np

from functools import partial
from typing import Callable, Union, List
from easydict import EasyDict as edict
from DeepLog import logger, Timer
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from TalkingFace.util import merge_from_two_image

current_pwd = os.getcwd()
logger.info(current_pwd)

from .get_disentangle_landmarks import DisentangledLandmarks, landmarks_visualization, draw_landmarks
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "ExpressiveVideoStyleGanEncoding"))

from ExpressiveEncoding.train import StyleSpaceDecoder, stylegan_path, from_tensor,\
                                     PoseEdit, get_detector, get_face_info, \
                                     gen_masks, to_tensor, imageio

from .module import BaseLinear
from .equivalent_offset import fused_offsetNet

psnr_func = lambda x,y: 20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - y) ** 2)))
#norm = lambda x: x
#norm = lambda x: np.clip(x / 100, -1, 1)

#norm = lambda x: ((x) / (x.max(axis = (0,1), keepdims = True)))
norm = lambda x: ((x - x.min(axis = (0,1), keepdims = True)) / (x.max(axis = (0,1), keepdims = True) - x.min(axis = (0,1), keepdims = True)) - 0.5) * 2

#norm = lambda x: x / 512
norm_torch = lambda x: ((x - x.amin(axis = (0,1), keepdim = True)) / (x.amax(axis = (0,1), keepdim = True) - x.amin(axis = (0,1), keepdim = True)) - 0.5) * 2

_linear = lambda x: x
_minmax = lambda x: (((x - x.amin(dim = (0,1), keepdim = True)) / (x.amax(dim = (0,1), keepdim = True) - x.amin(dim = (0,1), keepdim = True))) - 0.5) * 2
_normal = lambda x: x / 512
_minmax_inner = lambda x: (((x - x.amin(dim = (1), keepdim = True)) / (x.amax(dim = (1), keepdim = True) - x.amin(dim = (1), keepdim = True) + 1e-4)) - 0.5) * 2
_minmax_constant = lambda x, _min, _max: (torch.clip((x - _min) / (_max - _min), 0.0, 1.0) - 0.5) * 2
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
        self.pixel_range = list(range(pixel_range[0], pixel_range[1]))
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
        batchnorm = kwargs.get("batchnorm", True)
        is_refine = kwargs.get("is_refine", False)

        if depth > 0:
            _modules = [nn.Linear(in_channels, base_channels)] + ([torch.nn.Dropout1d(0.5)] if dropout else []) + \
                       [*([BaseLinear(base_channels, base_channels, skip = skip, batchnorm = batchnorm)] +  ([torch.nn.Dropout1d(0.5)] if dropout else []))] * depth + \
                       [*([nn.Linear(base_channels, out_channels)] + ([torch.nn.Dropout1d(0.5)] if dropout else []))]
        else:
            _modules = [nn.Linear(in_channels, out_channels)]#, nn.Linear(out_channels, out_channels)]

        self.net = nn.Sequential(*_modules)
        #self.act = nn.Tanh()

        self.refine = None

        #self.scale = nn.Parameters(torch.ones(in_channels)
        #self.direction = nn.Sequential(nn.Linear(base_channels, out_channels),torch.nn.Dropout1d(0.25), nn.Tanh())
        #self.intensity = nn.Sequential(nn.Linear(base_channels, out_channels),torch.nn.Dropout1d(0.25), nn.ReLU())

        norm_type = '_' + kwargs.get("norm_type", 'linear')
        norm_func = eval(norm_type)

        if norm_type == '_minmax_constant':
            if '_max' in kwargs and '_min' in kwargs:
                _min, _max = torch.tensor(kwargs['_min'], dtype = torch.float32), torch.tensor(kwargs['_max'], dtype = torch.float32)
            else:
                _min = torch.zeros((1,1,2))
                _max = torch.zeros((1,1,2))
            self.register_buffer('_min', _min)
            self.register_buffer('_max', _max)

        self.norm_type = norm_type
        self.norm = norm_func
        self.register_buffer("clip_values", torch.ones((out_channels, 2), dtype = torch.float32))

    def set_attr(self, clip_values: torch.Tensor):
        self.clip_values = clip_values

    def forward(self, x):
        
        if self.norm_type == '_minmax_constant':
            x = self.norm(x, self._min, self._max)
        else:
            x = self.norm(x)

        n = x.shape[0]
        x = x.reshape(n, -1)
        # y = self.act(self.net(x))
        y = self.net(x)
        """
        i = self.intensity(y)
        d = self.direction(y)
        return i * d
        """
        #return self.act(y)
        self.clip_values.to(x)
        return torch.clip(y, self.clip_values[:,0], self.clip_values[:,1])

        #return (0.5 * torch.clip(y, -1.0, 1.0) + 0.5) * (self.clip_values[:, 1] - self.clip_values[:, 0]) + self.clip_values[:, 0]

class Dataset:
    def __init__(
                 self,
                 attributes_path: str, # final file is a pt-format file.
                 ldm_path: str, # final file is a numpy-format file.
                 id_path: str,
                 id_landmark_path :str, 
                 augmentation: bool = False
                ):

        assert os.path.exists(attributes_path), f"attribute path {attributes_path} not exist."
        assert os.path.exists(ldm_path), f"attribute path {ldm_path} not exist."

        gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(id_path)

        self.attributes = torch.load(attributes_path)
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
        self.offsets = (landmarks - id_landmark)[:, 48:68, :]
        self.offset_range = self.get_offset_range(self.offsets)

        #self.offsets = norm(self.offsets)
        self.length = len(self.attributes) #round(len(self.attributes) * ratio)
        self.landmarks = landmarks
        current_folder = "/data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/"
        self.gen_files = [os.path.join(current_folder, x) for x in gen_file_list]

        self.augmentation = augmentation

        if augmentation:

            self.ops = [
                            shift(pixel_range = [-20, 20], dim = 'y')
                       ]

            logger.info([op for op in self.ops])
    def __len__(self):
        return self.length

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

    def get_offset_range(self, x):
        return np.min(x, axis = (0, 1), keepdims = True), np.max(x, axis = (0, 1), keepdims = True)

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


        if self.augmentation:
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
    #first 5 elements.
    for k, v in alphas[region_range[0]:region_range[1]]:
        for i in v:
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

    snapshots = os.path.join(save_path, "snapshots")
    tensorboard_path = os.path.join(save_path, "tensorboard", f"{time.time()}")
    os.makedirs(snapshots, exist_ok = True)
    os.makedirs(tensorboard_path, exist_ok = True)
    device = "cuda:0"
    writer = SummaryWriter(tensorboard_path)
    decoder = StyleSpaceDecoder(stylegan_path).to(device)
    for p in decoder.parameters():
        p.requires_grad = True

    datasets_list = []
    for _config in config.data:
        dataset_config = _config.dataset
        dataset = Dataset(os.path.join(current_pwd, dataset_config.attr_path),\
                          os.path.join(current_pwd,dataset_config.ldm_path),\
                          os.path.join(current_pwd, dataset_config.id_path), \
                          os.path.join(current_pwd, dataset_config.id_landmark_path), \
                          augmentation = False if not hasattr(dataset_config,"augmentation") else dataset_config.augmentation \
                          )
        datasets_list.append(dataset)
        max_values = dataset.get_attr_max()
        offset_range = dataset.offset_range
    dataset = torch.utils.data.ConcatDataset(datasets_list)
    dataloader = DataLoader(dataset, batch_size = config.batchsize, shuffle = True, num_workers = 8)

    dataset_config = config.val
    val_dataset = Dataset(os.path.join(current_pwd, dataset_config.attr_path), os.path.join(current_pwd,dataset_config.ldm_path), os.path.join(current_pwd, dataset_config.id_path), os.path.join(current_pwd, dataset_config.id_landmark_path))
    val_dataloader = DataLoader(val_dataset, batch_size = config.batchsize, shuffle = False)

    # init net
    net_config = config.net
    is_refine = False if not hasattr(net_config, "is_refine") else net_config.is_refine

    net = offsetNet(\
                    net_config.in_channels * 2, size_of_alpha, \
                    net_config.depth, \
                    base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , \
                    dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                    batchnorm = True if not hasattr(net_config, "batchnorm") else net_config.batchnorm, \
                    skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                    is_refine = is_refine, \
                    norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                    _min = offset_range[0], \
                    _max = offset_range[1]
                   ) 

    best_nme = 100.0
    best_acc = 0.0
    if resume_path is not None:
        logger.info(f"resume training from {resume_path}")
        state_dict = torch.load(resume_path)
        weight = state_dict
        if "best_value" in state_dict :
            weight = state_dict['weight']
            best_nme = state_dict['best_value']
            logger.info(f"load weight and nme : {best_nme}")
        net.load_state_dict(weight, False)
    net.set_attr(max_values)
    net.to(device)


    # enable calculate derivate.
    for p in net.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(net.parameters(), lr = net_config.lr)
    sche = torch.optim.lr_scheduler.StepLR(optimizer, config.optim.step, gamma = config.optim.gamma)
    #loss = torch.nn.SmoothL1Loss()
    #loss = torch.nn.CosineSimilarity()
    #loss = torch.nn.MSELoss()
    #loss_image = torch.nn.L1Loss()
    #loss_fea = torch.nn.MSELoss()

    #loss_fea = SmoothL1LossMyself()
    #loss_fea = torch.nn.CosineSimilarity()

    #loss_d = torch.nn.CrossEntropyLoss()
    loss_d = torch.nn.CosineSimilarity(dim = 0)
    loss_i = torch.nn.MSELoss()
    #loss_i = torch.nn.L1Loss()

    start_epoch = 1 if not hasattr(config, "start_epoch") else config.start_epoch
    pbar = tqdm(range(start_epoch, config.epochs + start_epoch))
    total_count = 0
    last_path = None
    for epoch in pbar:
        for idx, data in enumerate(dataloader):
            attr, offset = data
            attr = attr.to(device)
            offset = offset.to(device)
            d_loss_value = 0.0
            pred_attr = net(offset)

            attr = (attr - net.clip_values[:, 0]) / (net.clip_values[:, 1] - net.clip_values[:, 0])
            pred_attr = (pred_attr - net.clip_values[:, 0]) / (net.clip_values[:, 1] - net.clip_values[:, 0])

            d_loss_value = 1 - loss_d(pred_attr, attr).mean() #(1 - loss_d(torch.sign(pred_attr), torch.sign(attr)).mean())
            i_loss_value = loss_i(attr, pred_attr)
             #loss_d(torch.sigmoid(pred_attr), (attr >= 0).to(torch.float32)) #(1 - loss_d(torch.sign(pred_attr), torch.sign(attr)).mean())

            loss_value = i_loss_value * 1.0 + d_loss_value * 0.0
            loss_value.backward()
            optimizer.step()
            total_count += 1

            if idx % config.show_internal == 0:
                logger.info(f"epoch:{epoch}: {idx+1}/{len(dataloader)} loss {loss_value.mean().item()}, d_loss {d_loss_value.mean().item()} i_loss {i_loss_value.mean().item()} ")
                writer.add_scalar("loss", loss_value.mean().item(), total_count)

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

            #pred_attr = (pred_attr * (net.clip_values[:, 1] - net.clip_values[:, 0])) + net.clip_values[:, 0]
            acc_value += (torch.sign(pred_attr) == torch.sign(attr)).to(torch.float32).mean()
            attr = (attr - net.clip_values[:, 0]) / (net.clip_values[:, 1] - net.clip_values[:, 0])
            pred_attr = (pred_attr - net.clip_values[:,0]) / (net.clip_values[:,1] - net.clip_values[:,0])

            nme_value += torch.nn.functional.mse_loss(attr, pred_attr).mean()
            sim_value += torch.nn.functional.cosine_similarity(pred_attr, attr).mean()
        nme_value /= len(val_dataloader)
        acc_value /= len(val_dataloader)
        sim_value /= len(val_dataloader)
        rnme_value = torch.sqrt(nme_value).item()
        nme_value = nme_value.item()
        acc_value = acc_value.item()
        sim_value = sim_value.item()
        logger.info(f"nme is {nme_value}, acc is {acc_value}, sim is {sim_value} rnme is {rnme_value}.")
        writer.add_scalar("nme", nme_value ,global_step = epoch)
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
                      e4e_latent_path: str,
                      save_path: str,
                      video_landmark_path: str,
                      driving_images_dir: str = None,
                      **kwargs
                     ):


    assert save_path.endswith('mp4'), "expected postfix mp4, but got is {save_path}."

    device = "cuda:0"
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    net_config = config.net
    net = offsetNet(\
                    net_config.in_channels * 2, size_of_alpha, \
                    net_config.depth, \
                    base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , \
                    dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                    batchnorm = True if not hasattr(net_config, "batchnorm") else net_config.batchnorm, \
                    skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                    norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                   )
    logger.info(net)
    net.to(device)
    state_dict = torch.load(offset_weight_path)
    weight = state_dict
    if "best_value" in state_dict :
        weight = state_dict['weight']
        best_nme = state_dict['best_value']
        logger.info(f"load weight and nme : {best_nme}")

    net.load_state_dict(weight, False)

    logger.info(net._min)
    logger.info(net._max)
    net.eval()
    resolution = kwargs.get("resolution", 1024)
    decoder = StyleSpaceDecoder(stylegan_path, to_resolution = resolution)
    decoder.load_state_dict(torch.load(pti_weight_path), False)
    decoder.to(device)
    attributes = torch.load(attributes_path)
    get_disentangled_landmarks = DisentangledLandmarks()
    _, selected_id_image, selected_id_latent, selected_id = torch.load(e4e_latent_path)
    if isinstance(landmarks, str):
        landmarks = np.load(landmarks)[...,:2]
    # hard code id video landmarks.
    id_landmarks = np.load(video_landmark_path)[selected_id - 1:selected_id, :]
    #id_landmarks = get_disentangled_landmarks(np.uint8(selected_id_image))
    #landmark_offsets = norm((landmarks - id_landmarks)[:, 48:68, :])
    landmark_offsets = (landmarks - id_landmarks)[:, 48:68, :]

    driving_files = None
    if driving_images_dir is not None:
        assert os.path.isdir(driving_images_dir), "expected directory."
        driving_files = sorted(os.listdir(driving_images_dir), key = lambda x: int(''.join(re.findall('[0-9]+', x))))
        driving_files = [os.path.join(driving_images_dir, x) for x in driving_files]

    logger.info(landmark_offsets.shape)
    n = min(landmark_offsets.shape[0], len(attributes), len(os.listdir(pose_latent_path)))
    t = Timer()
    p_bar = tqdm.tqdm(range(n))
    with imageio.get_writer(save_path, fps = 25) as writer:
        for i in p_bar:
            attribute = attributes[i]
            w_plus_with_pose = torch.load(os.path.join(pose_latent_path, f'{i + 1}.pt'))
            #w_plus_with_pose = torch.load(os.path.join(pose_latent_path, f'{1}.pt'))
            style_space_latent = decoder.get_style_space(w_plus_with_pose)
            landmark_offset = torch.from_numpy(landmark_offsets[i]).unsqueeze(0).float().to(device)
            #ss_updated = update_region_offset_v2(style_space_latent, torch.tensor(attribute[1]).reshape(1, -1).to(device), [0, len(alphas)])
            ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to(device), [8, len(alphas)])
            with torch.no_grad():
                offset = net(landmark_offset)
            ss_updated = update_lip_region_offset(ss_updated, offset, version = 'v2')
            #ss_updated = torch.load(os.path.join(pose_latent_path, f'{i + 1}.pt'))
            #ss_updated = [x.to(device) for x in ss_updated]
            output = np.clip(from_tensor(decoder(ss_updated) * 0.5 + 0.5) * 255.0, 0.0, 255.0)
            h,w = output.shape[:2]

            if 0:
                # facial core latent.
                output_core_latent = np.clip(from_tensor(decoder(style_space_latent) * 0.5 + 0.5) * 255.0, 0.0, 255.0)
                output = np.concatenate((output, output_core_latent), axis = 0)

                # facial reenactment.
                ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1]).reshape(1, -1).to(device), [0, len(alphas)])
                output_reenactment = np.clip(from_tensor(decoder(ss_updated) * 0.5 + 0.5) * 255.0, 0.0, 255.0)
                output = np.concatenate((output, output_reenactment), axis = 0)

            if driving_files is not None:
                try:
                    image = cv2.imread(driving_files[i])[...,::-1]
                    #image = cv2.imread(driving_files[0])[...,::-1]
                except Exception as e:
                    break
                w = h = 512
                image = cv2.resize(image, (w,h))
                output = cv2.resize(output, (w,h))
                output = merge_from_two_image(image, output)
                output = np.concatenate((output, image), axis = 0)
            writer.append_data(np.uint8(output))
