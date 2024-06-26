"""blender with origin video.
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
import click
import time

import torch.nn as nn
import numpy as np

from functools import partial
from typing import Callable, Union, List
from easydict import EasyDict as edict
from PIL import Image
from functools import partial

from torchvision import transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter

from DeepLog import logger

cwd = os.getcwd()

from .util import make_dataset
from .infer import infer


LOG2 = lambda x: np.log(x) / np.log(2)

class Sampler(nn.Module):
    def __init__(
                 self,
                 sample_rate: float
                ):
        super().__init__()

        self.sampler = partial(nn.functional.interpolate, scale_factor = sample_rate, mode = 'nearest')

    def forward(self, x):
        return self.sampler(x)

class BNBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                sampler: float
                ):
        super(BNBlock, self).__init__()
        self.block = nn.Sequential(
                                   Sampler(sampler),
                                   nn.Conv2d(in_channels, out_channels, 3, 1, 1), 
                                   nn.BatchNorm2d(out_channels), 
                                   nn.LeakyReLU(),
                                   nn.Conv2d(out_channels, out_channels, 3, 1, 1), 
                                   nn.BatchNorm2d(out_channels), 
                                   nn.LeakyReLU(),
                                  )


    def forward(self, x):
        return self.block(x)

class BlenderNet(nn.Module):
    """ simple Unet.
    """
    def __init__(
                 self, 
                 in_channels: int,
                 multiplier: int = 2,
                 base_channels: int = 64,
                 size: int = 512
                 ):
        super(BlenderNet, self).__init__()

        _channels_dict = {
                           512: base_channels,
                           256: base_channels * 2,
                           128: base_channels * 4,
                           64: base_channels * 8
                        }
        layers = int(LOG2(size) - LOG2(64) + 1)

        _size_list = [64 * 2 ** i for i in range(0, layers)]

        modules_enc = []
        modules_dec = []
        conv0 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        final = nn.Sequential(nn.Conv2d(base_channels, 2, 1, 1), nn.Tanh())

        in_size = _channels_dict[512]
        for _size in _size_list[::-1][1:]:
            out_size = _channels_dict[_size]
            modules_enc += [BNBlock(in_size, out_size, 0.5)]
            in_size = out_size
    
        in_size = out_size
        for _size in _size_list[1:]:
            out_size = _channels_dict[_size]
            modules_dec += [BNBlock(in_size, out_size, 2)]
            in_size = out_size
    
        self.module = nn.Sequential(conv0, *modules_enc, *modules_dec, final)

    def forward(self, x, y):
        n, _, h, w = x.shape
        offset = self.module(torch.cat((x,y), dim = 1))
        i, j = torch.meshgrid(torch.Tensor(list(range(w))), torch.Tensor(list(range(h))), indexing = "xy")

        i = ((i / (w -1)) - .5) / 0.5
        j = ((j / (h -1)) - .5) / 0.5
        grid = torch.stack((i, j), dim = 2).view(1, h, w, 2).repeat(n, 1, 1, 1).to(x) + offset.permute(0, 2, 3, 1)
        return nn.functional.grid_sample(x, grid)

class Dataset:
    def __init__(
                 self, 
                 root_path
                 ):

        assert os.path.exists(root_path), f"{root_path} not exists."
        
        # gt_image
        gt_path = os.path.join(root_path, "gt")
        self.gt_paths = sorted(make_dataset(gt_path), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))

        # gen_image
        self.gen_path = os.path.join(root_path, "gen")

        # mask image
        self.mask_path = os.path.join(root_path, "mask")

        self.transform_gt = transforms.Compose(
                                            [
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]
                                          )


    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        _,gt_path = self.gt_paths[idx]
        file_name = os.path.basename(gt_path)
        gt_image = cv2.imread(gt_path)[..., ::-1].copy()
        """
        gen_image = cv2.imread(os.path.join(self.gen_path, file_name))[..., ::-1]
        """
        mask = cv2.imread(os.path.join(self.mask_path, file_name), 0)

        gt = self.transform_gt(gt_image)
        mask = transforms.ToTensor()(mask)

        random_choice = random.random()
        gen = gt.clone()
        if random_choice < 0.4:
            gen = transforms.RandomAffine(degrees = 2, translate = (0.01, 0.02), shear = 1)(gt)
        if random_choice >= 0.4 and random_choice < 0.8:
            gen = transforms.RandomPerspective(0.1, 1)(gt)
        return gt, gen, mask

def get_dataloader(
                    root_path,
                    batch_size
                  ):

    dataset = Dataset(root_path)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 8)


@click.command()
@click.option("--config_path", default = None)
@click.option("--resume_path", default = None)
@click.option("--save_path", default = None)
def trainer(
            config_path,
            resume_path,
            save_path
            ):

    tensorboard_path = os.path.join(cwd, save_path, "tensorboard")
    snapshots_path = os.path.join(cwd, save_path, "snapshots")
    eval_path = os.path.join(cwd, save_path, "eval")

    os.makedirs(tensorboard_path, exist_ok = True)
    os.makedirs(snapshots_path, exist_ok = True)
    os.makedirs(eval_path, exist_ok = True)
    
    writer = SummaryWriter(os.path.join(tensorboard_path, str(time.time())))

    with open(config_path, 'r') as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    dataset = config.dataset
    dataloader = get_dataloader(os.path.join(cwd,dataset.path), dataset.batch_size)

    net_config = config.net

    net = BlenderNet(net_config.in_channels)
    device = "cuda:0"
    net.to(device)
    net.train()

    start_epoch = 1
    epochs = config.epochs

    if resume_path is not None:
        state_dict = torch.load(resume_path)
        score = state_dict["score"]
        epoch = state_dict["epoch"]
        logger.info(f"resume from {resume_path}.model score is {score}. start epoch {epoch}")
        net.load_state_dict(state_dict["weight"])
        start_epoch = epoch

    logger.info("optimizer initialization.")
    optimizer = torch.optim.Adam(net.parameters(), lr = net_config.lr)
    sche = torch.optim.lr_scheduler.StepLR(optimizer, net_config.step, gamma = net_config.gamma)
    
    loss_l2 = torch.nn.MSELoss()

    pbar = tqdm.tqdm(range(start_epoch, epochs + 1))

    total_iter = 0

    min_loss = 0xffff
    for epoch in pbar:
        loss_mean = 0.0
        counter = 0
        for idx, data in enumerate(dataloader):
            
            gt, gen, mask = data   
            gt = gt.to(device)
            gen = gen.to(device)
            mask = mask.to(device)
            fake = net(gen, gt)
            local_loss = loss_l2(fake * mask, gt * mask)
            global_loss = loss_l2(fake, gt)
            loss = local_loss + global_loss
            loss.backward()
            optimizer.step()

            if idx % config.internal == 0:
                logger.info(f"{idx}/{epoch}/{epochs}: loss(local) {local_loss.item()} loss(global) {global_loss.item()}. ")
                writer.add_scalar("loss", loss.item(), total_iter)
                images_in_training = torch.cat((fake, fake * mask, gt * mask, gen * mask), dim =2)
                writer.add_image('image', make_grid(images_in_training.detach(),normalize=True, scale_each=True), total_iter)
                loss_mean += loss.item()
                counter += 1
            total_iter += 1
        
        loss_mean /= counter
        if min_loss > loss_mean:
            logger.info(f"min loss: {loss_mean}. save checkpoint {epoch}.pth.")
            torch.save(
                        dict(
                                score = loss_mean,
                                epoch = epoch,
                                weight = net.state_dict()
                            ),
                        os.path.join(snapshots_path, f"{epoch}.pth")
                      )
    
            min_loss = loss_mean

        if epoch % config.eval_internal == 0:
            net.eval()

            def blender(x, y, _m):
                def to_tensor(_x):
                    return torch.from_numpy(_x).unsqueeze(0).permute((0,3,1,2)) / 255.0

                norm = lambda x: (x - 0.5) / 0.5
                x = norm(to_tensor(x))
                y = norm(to_tensor(y))
                _m = to_tensor(_m)

                output = net(x.to(device), y.to(device))
                return ((output.detach().squeeze().permute((1,2,0)).cpu().numpy() * 0.5) + 0.5) * 255.0

            infer(
                   os.path.join(cwd, config.eval.config_path),
                   os.path.join(eval_path, f"{epoch}.mp4"),
                   blender = blender,
                   frames = 100
                 )

            net.train()

if __name__ == "__main__":
    trainer()
