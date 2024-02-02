"""This script is the test script for Deep3DFaceRecon_pytorch
Pytorch Deep3D_Recon is 8x faster than TF-based, 16s/iter ==> 2s/iter
"""

import os
# os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" + os.path.abspath("deep_3drecon")
import torch
import torch.nn as nn
# from .deep_3drecon_models.facerecon_model import FaceReconModel
from .deep_3drecon_models.facerecon_model_zxy import FaceReconModel
from .util.preprocess import align_img,align_img_param, align_img_coeff_and_param
from PIL import Image
import numpy as np
from .util.load_mats import load_lm3d
import torch 
import pickle as pkl
from PIL import Image

from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
with open(f"deep_3drecon/reconstructor_opt.pkl", "rb") as f:
    opt = pkl.load(f) 
    
class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FaceReconModel(opt)
        self.model.setup(opt)
        self.model.device = 'cuda:0'
        self.model.parallelize()
        # self.model.to(self.model.device)
        self.model.eval()
        self.lm3d_std = load_lm3d(opt.bfm_folder) 
    
    def preprocess_data(self, im, lm, lm3d_std):
        # to RGB 
        H,W,_ = im.shape
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        _, im, lm, _ = align_img(Image.fromarray(convert_to_np(im)), convert_to_np(lm), convert_to_np(lm3d_std))
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
        return im, lm

    def preprocess_data_coeff_and_param(self, im, lm, lm3d_std):
        # to RGB 
        H,W,_ = im.shape
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        _, im, lm, _, params= align_img_coeff_and_param(Image.fromarray(convert_to_np(im)), convert_to_np(lm), convert_to_np(lm3d_std))
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
        return im, lm, params

    def preprocess_data_param(self, im, lm, lm3d_std):
        # to RGB 
        H,W,_ = im.shape
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        param = align_img_param(Image.fromarray(convert_to_np(im)), convert_to_np(lm), convert_to_np(lm3d_std))
        return param

    @torch.no_grad()
    def recon_coeff(self, batched_images, batched_lm5, return_image=True, batch_mode=True):
        bs = batched_images.shape[0]
        data_lst = []
        for i in range(bs):
            img = batched_images[i]
            lm5 = batched_lm5[i]
            align_im, lm = self.preprocess_data(img, lm5, self.lm3d_std)
            data = {
                'imgs': align_im,
                'lms': lm
            }
            data_lst.append(data)
        if not batch_mode:
            coeff_lst = []
            align_lst = []
            for i in range(bs):
                data = data_lst
                self.model.set_input(data)  # unpack data from data loader
                self.model.forward()
                pred_coeff = self.model.output_coeff.cpu().numpy()
                align_im = (align_im.squeeze().permute(1,2,0)*255).int().numpy().astype(np.uint8)
                coeff_lst.append(pred_coeff)
                align_lst.append(align_im)
            batch_coeff = np.concatenate(coeff_lst)
            batch_align_img = np.stack(align_lst) # [B, 257]
        else:
            imgs = torch.cat([d['imgs'] for d in data_lst])
            lms = torch.cat([d['lms'] for d in data_lst])
            data = {
                'imgs': imgs,
                'lms': lms
            }
            self.model.set_input(data)  # unpack data from data loader
            self.model.forward()
            batch_coeff = self.model.output_coeff.cpu().numpy()
            batch_align_img = (imgs.permute(0,2,3,1)*255).int().numpy().astype(np.uint8)
        return batch_coeff, batch_align_img

    @torch.no_grad()
    def recon_coeff_and_trans(self, batched_images, batched_lm5, return_image=True, batch_mode=True):
        bs = batched_images.shape[0]
        data_lst = []
        params = []
        for i in range(bs):
            img = batched_images[i]
            lm5 = batched_lm5[i]
            align_im, lm, param = self.preprocess_data_coeff_and_param(img, lm5, self.lm3d_std)
            data = {
                'imgs': align_im,
                'lms': lm
            }
            data_lst.append(data)
            params.append(param)
        if not batch_mode:
            coeff_lst = []
            align_lst = []
            for i in range(bs):
                data = data_lst
                self.model.set_input(data)  # unpack data from data loader
                self.model.forward()
                pred_coeff = self.model.output_coeff.cpu().numpy()
                align_im = (align_im.squeeze().permute(1,2,0)*255).int().numpy().astype(np.uint8)
                coeff_lst.append(pred_coeff)
                align_lst.append(align_im)
            batch_coeff = np.concatenate(coeff_lst)
            batch_align_img = np.stack(align_lst) # [B, 257]
        else:
            imgs = torch.cat([d['imgs'] for d in data_lst])
            lms = torch.cat([d['lms'] for d in data_lst])
            data = {
                'imgs': imgs,
                'lms': lms
            }
            self.model.set_input(data)  # unpack data from data loader
            self.model.forward()
            batch_coeff = self.model.output_coeff.cpu().numpy()
            batch_align_img = (imgs.permute(0,2,3,1)*255).int().numpy().astype(np.uint8)
        params = np.stack(params)
        return batch_coeff, batch_align_img, params

    @torch.no_grad()
    def recon_coeff_param(self, batched_images, batched_lm5, return_image=True, batch_mode=True):
        bs = batched_images.shape[0]
        data_lst = []
        for i in range(bs):
            img = batched_images[i]
            lm5 = batched_lm5[i]
            param = self.preprocess_data_param(img, lm5, self.lm3d_std)
            data_lst.append(param)
        param = np.stack(data_lst)
        return param
    
    # todo: batch-wise recon!
        
    def forward(self, batched_images, batched_lm5, return_image=True):
        return self.recon_coeff(batched_images, batched_lm5, return_image)


    
