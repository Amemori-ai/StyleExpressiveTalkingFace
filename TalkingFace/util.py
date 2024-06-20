"""merge with original image
"""
import os
import cv2
import yaml
import re
import numpy as np

DEBUG = os.environ.get("DEBUG", True)
DEBUG = True if DEBUG in ["True", "TRUE", True, 1] else False

with open(os.path.join("/data1/wanghaoran/Amemori", "template.yaml")) as f:
    config = yaml.load(f, Loader = yaml.CLoader)

regions = eval(config["soft_mask_region"])
output_copy_region = eval(config["output_copy_region"])

from .ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.utils import make_dataset

def get_soft_mask_by_region():
    soft_mask  = np.zeros((512,512,3), np.float32)
    for region in regions:
        y1,y2,x1,x2 = region
        soft_mask[y1:512,x1:x2,:]=1
    #soft_mask = cv2.GaussianBlur(soft_mask, (101, 101), 11)
    soft_mask = soft_mask.astype(np.float32)
    return soft_mask

def merge_from_two_image(
                         master: np.ndarray,
                         slave: np.ndarray,
                         mask: np.ndarray = None,
                         blender: object = None
                        ) -> np.ndarray:
    master = master.astype(np.float32)
    slave = slave.astype(np.float32)
    if mask is not None:
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        mask = np.uint8(mask)

        # dilate
        erosion_size = 15
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
        mask_erosion = cv2.erode(mask, element)

        dilate_size = 5
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_size + 1, 2 * dilate_size + 1), (dilate_size, dilate_size))
        mask_dilate = cv2.dilate(mask, element)
        mask_diff = (mask_dilate - mask_erosion)[..., np.newaxis]
        #mask_diff = mask - mask_erosion[..., np.newaxis]

        if blender is not None:
            output = blender(master, slave, np.uint8(mask_diff * 255))
            mask_diff = cv2.GaussianBlur(mask_diff, (101, 101), 11)
            if mask_diff.ndim < 3:
                mask_diff = mask_diff[..., np.newaxis]
            slave = output * mask_diff + (1 - mask_diff) * slave
        else:
            pass
            #mask_diff = cv2.boxFilter(mask_diff.astype(np.float32), -1, ksize = (21, 21))
            #if mask_diff.ndim < 3:
            #    mask_diff = mask_diff[..., np.newaxis]
            #slave = master * mask_diff + (1 - mask_diff) * slave
            #mask = cv2.GaussianBlur(mask, (101, 101), 11)
            #if mask.ndim < 3:
            #    mask = mask[..., np.newaxis]
        mask = cv2.boxFilter(mask_erosion.astype(np.float32), -1, ksize = (21, 21))
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        merge_mask = mask
    else:
        merge_mask = np.zeros_like(master)
        dilate_erode_mask = merge_mask.copy()
        for (y1,y2, x1, x2) in output_copy_region:
            merge_mask[y1:y2, x1:x2, ...] = 1.0
        pad_l = 5
        pad_r = 20
        dilate_erode_mask[y1 - pad_l: y2 + pad_r, x1 - pad_l : x2 + pad_r, ...] = 1.0
        mask_diff = dilate_erode_mask - merge_mask
        mask_diff = cv2.boxFilter(mask_diff.astype(np.float32), -1, ksize = (21, 21))
        if mask_diff.ndim < 3:
            mask_diff = mask_diff[..., np.newaxis]
        slave = master * mask_diff + (1 - mask_diff) * slave
    
    output = merge_mask * slave + (1 - merge_mask) * master
    partial_mask = get_soft_mask_by_region()
    #output = output * partial_mask + master * (1 - partial_mask)

    if DEBUG:
        cv2.imwrite("merge_mask.jpg", merge_mask * master)
        cv2.imwrite("partial_mask.jpg", partial_mask * master * merge_mask)

    return np.clip(output, 0.0, 255.0).astype(np.uint8)
