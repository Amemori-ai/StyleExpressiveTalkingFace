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
                         slave: np.ndarray
                        ) -> np.ndarray:
    master = master.astype(np.float32)
    slave = slave.astype(np.float32)

    merge_mask = np.zeros_like(master)
    for (y1,y2, x1, x2) in output_copy_region:
        merge_mask[y1:y2, x1:x2, ...] = 1.0
    
    output = merge_mask * slave + (1 - merge_mask) * master
    partial_mask = get_soft_mask_by_region()
    #output = output * partial_mask + master * (1 - partial_mask)

    if DEBUG:
        cv2.imwrite("merge_mask.jpg", merge_mask * master)
        cv2.imwrite("partial_mask.jpg", partial_mask * master * merge_mask)

    return np.clip(output, 0.0, 255.0).astype(np.uint8)
