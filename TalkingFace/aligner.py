import os 
import sys
import cv2
import yaml
import torch

import torch.nn as nn
import numpy as np

from typing import Callable, Union
from easydict import EasyDict as edict
from DeepLog import logger
from torchvision import transforms
from torch.utils.data import DataLoader


from .get_disentangle_landmarks import DisentangledLandmarks
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

class offsetNet(nn.Module):
    def __init__(
                 self, 
                 in_channels,
                 out_channels,
                 depth = 0,
                 base_channels = 512
                ):
        super().__init__()
        if depth > 0:
            class BaseLiner(nn.Module):
                def __init__(self,
                             input_channels,
                             output_channels
                             ):
                    super().__init__()
                    self.module = nn.Sequential(*[nn.Linear(input_channels,output_channels), nn.LeakyReLU()])
                def forward(self, x):
                    return self.module(x)
            _modules = [nn.Linear(in_channels, base_channels)] + \
                       [BaseLiner(base_channels, base_channels)] * depth + \
                       [nn.Linear(base_channels, out_channels)]
                       
        else:
            _modules = [nn.Linear(in_channels, out_channels)]

        self.net = nn.Sequential(*_modules)

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape(n, -1)
        return self.net(x)

class Dataset:
    def __init__(
                 self,
                 attributes_path: str, # final file is a pt-format file.
                 ldm_path: str, # final file is a numpy-format file.
                 selected_id: int
                ):

        assert os.path.exists(attributes_path), f"attribute path {attributes_path} not exist."
        assert os.path.exists(ldm_path), f"attribute path {ldm_path} not exist."
        assert selected_id > 0, "selected id must be more than zero."

        attributes = torch.load(attributes_path)
        landmarks = np.load(ldm_path)

        #assert len(attributes) == len(landmarks), "attributes length unequal with landmarks."

        self.attributes = attributes
        self.offsets = landmarks - landmarks[selected_id - 1: selected_id, ...]

    def __len__(self):
        return len(self.attributes)

    def __getitem__(
                    self,
                    index
                   ):
        return self.attributes[index][0], torch.stack(self.attributes[index][1][:size_of_alpha], dim = 0), torch.from_numpy(self.offsets[index][48:68, :]).to(torch.float32) / 512

def update_lip_region_offset(
                              dlatents,
                              offset
                           ):
    return update_region_offset(dlatents, offset, [0,8])

def update_region_offset(
                          dlatents,
                          offset,
                          region_range
                        ):
    dlatents_tmp = [latent.clone() for latent in dlatents]
    count = 0
    # first 5 elements.
    for k, v in alphas[region_range[0]:region_range[1]]:
        for i in v:

            dlatents_tmp[k][:, i] = dlatents[k][:, i] + offset[:,count]
            count += 1
    return dlatents_tmp

def aligner(
            config_path: str,
            save_path: str
           ):

    import time
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    where_am_i = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(where_am_i, "ExpressiveVideoStyleGanEncoding"))

    from ExpressiveEncoding.train import StyleSpaceDecoder, stylegan_path, from_tensor,\
                                         PoseEdit, get_detector, get_face_info, \
                                         gen_masks
    from torchvision.utils import make_grid

    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    snapshots = os.path.join(save_path, "snapshots")
    tensorboard_path = os.path.join(save_path, "tensorboard", f"{time.time()}")
    os.makedirs(snapshots, exist_ok = True)
    os.makedirs(tensorboard_path, exist_ok = True)
    device = "cuda:0"
    writer = SummaryWriter(tensorboard_path)
    
    data_config = config.data

    decoder = StyleSpaceDecoder(stylegan_path).to(device)
    for p in decoder.parameters():
        p.requires_grad = True

    _, selected_id_image, selected_id_latent, selected_id = torch.load(data_config.id_path)
    pose_edit = PoseEdit()
    detector = get_detector()
    face_info = get_face_info(cv2.cvtColor(np.uint8(selected_id_image), cv2.COLOR_BGR2RGB), detector)
    pitch = face_info.pitch
    yaw = face_info.yaw

    with torch.no_grad():
        zflow = pose_edit(selected_id_latent, yaw, pitch)

    dataset = Dataset(data_config.attr_path, data_config.ldm_path, selected_id)
    dataloader = DataLoader(dataset, batch_size = config.batchsize, shuffle = True)

    # init net
    net_config = config.net
    net = offsetNet(net_config.in_channels * 2, size_of_alpha, net_config.depth)
    net.to(device)

    # enable calculate derivate.
    for p in net.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(net.parameters(), lr = net_config.lr)

    #loss = torch.nn.SmoothL1Loss()
    #loss = torch.nn.CosineSimilarity()
    loss = torch.nn.MSELoss()
    pbar = tqdm(range(1, config.epochs + 1))
    total_count = 0
    last_path = None
    for epoch in pbar:
        for idx, data in enumerate(dataloader):
            pose, attr, offset = data
            attr = attr.to(device)
            offset = offset.to(device)
            attr_pred = net(offset)
            yaw, pitch = pose[:,0], pose[:,1]
            n = attr.shape[0]
            with torch.no_grad():
                w_plus_with_pose = pose_edit(zflow.repeat(n,1,1), yaw, pitch, True)
            selected_id_ss = decoder.get_style_space(w_plus_with_pose)

            gt_tensor = get_gen_image(selected_id_ss, attr, decoder)
            pred_tensor = get_gen_image(selected_id_ss, attr_pred, decoder)
            loss_value = loss(pred_tensor, gt_tensor) #(1 - loss(attr_pred, attr).mean())
            loss_value.backward()
            optimizer.step()
            total_count += 1
            if idx % config.show_internal == 0:
                logger.info(f"epoch:{epoch}: {idx+1}/{len(dataloader)} loss {loss_value.mean().item()}")
                writer.add_scalar("loss", loss_value.mean().item(), total_count)
                image_to_show = torch.cat((pred_tensor, gt_tensor),dim = 2)
                writer.add_image(f'image', make_grid(image_to_show.detach(),normalize=True, scale_each=True), total_count)

        if epoch % config.save_internal == 0:
            last_path =  os.path.join(snapshots, f"{epoch}.pth")
            torch.save(net.state_dict(), last_path)
    return last_path

def get_gen_image(
                    id_ss_latent: torch.tensor,
                    offset: list, 
                    decoder: Callable 
                 ):

    ss_latent = update_lip_region_offset(id_ss_latent, offset)
    return decoder(ss_latent)

def sync_lip_validate(
                      landmarks: Union[str , np.ndarray],
                      config_path: str,
                      offset_weight_path: str,
                      pti_weight_path: str,
                      attributes_path: str,
                      e4e_latent_path: str,
                      save_path: str
                     ):

    where_am_i = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(where_am_i, "ExpressiveVideoStyleGanEncoding"))
    import imageio
    from ExpressiveEncoding.train import StyleSpaceDecoder, stylegan_path, from_tensor, \
                                         PoseEdit, get_face_info, get_detector

    assert save_path.endswith('mp4'), "expected postfix mp4, but got is {save_path}."

    device = "cuda:0"
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    net_config = config.net
    net = offsetNet(net_config.in_channels * 2, size_of_alpha, net_config.depth)
    net.to(device)
    state_dict = torch.load(offset_weight_path)
    net.load_state_dict(state_dict)
    
    decoder = StyleSpaceDecoder(stylegan_path)
    decoder.load_state_dict(torch.load(pti_weight_path))
    pose_edit = PoseEdit()

    attributes = torch.load(attributes_path)
    get_disentangled_landmarks = DisentangledLandmarks()
    _, selected_id_image, selected_id_latent, selected_id = torch.load(e4e_latent_path)
    if isinstance(landmarks, str):
        landmarks = np.load(landmarks)
    id_landmarks = get_disentangled_landmarks(np.uint8(selected_id_image))
    landmark_offsets = (landmarks - id_landmarks)[:, 48:68, :]


    detector = get_detector()
    face_info = get_face_info(cv2.cvtColor(np.uint8(selected_id_image), cv2.COLOR_BGR2RGB), detector)
    pitch = face_info.pitch
    yaw = face_info.yaw

    with torch.no_grad():
        zflow = pose_edit(selected_id_latent, yaw, pitch)

    n = min(landmark_offsets.shape[0], len(attributes))
    with imageio.get_writer(save_path, fps = 25) as writer:
        for i in range(n):
            attribute = attributes[i]
            yaw, pitch = attribute[0]
            with torch.no_grad():
                w_plus_with_pose = pose_edit(zflow, yaw, pitch, True)

            style_space_latent = decoder.get_style_space(w_plus_with_pose)
            landmark_offset = torch.from_numpy(landmark_offsets[i]).unsqueeze(0).float().to(device)
            #ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to(device), [8, len(alphas)])
            #offset = net(landmark_offset/512.0)
            #ss_updated = update_lip_region_offset(ss_updated, offset)
            ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1]).reshape(1, -1).to(device), [0, len(alphas)])
            output = from_tensor(decoder(ss_updated) * 0.5 + 0.5) * 255.0
            writer.append_data(np.uint8(output))
