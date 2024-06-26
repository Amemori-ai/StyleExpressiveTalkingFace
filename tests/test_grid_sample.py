import os
import sys

import torch
import cv2


def test_grid_sample():

    path = "/data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/results/exp010/0/data/smooth/0.png"
    image = cv2.imread(path)
    image = torch.from_numpy(image).permute((2,0,1)).unsqueeze(0).float()

    n,c,h,w = image.shape
    i, j = torch.meshgrid(torch.Tensor(list(range(w))), torch.Tensor(list(range(h))), indexing = "xy")
    #i = i / (w - 1)
    #j = j / (h - 1)
    i = ((i / (w -1)) - .5) / 0.5
    j = ((j / (h -1)) - .5) / 0.5
    print(torch.stack((i, j), dim = 2).shape)
    grid = torch.stack((i, j), dim = 2).view(1, h, w, 2).repeat(n, 1, 1, 1)
    image_resample = torch.nn.functional.grid_sample(image, grid)
    image_resample = image_resample.detach().squeeze().permute((1,2,0)).numpy()
    cv2.imwrite("resample.jpg", image_resample)

    


