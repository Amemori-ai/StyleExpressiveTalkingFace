import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import numpy as np


from TalkingFace.aligner import face_parsing
face_parse = face_parsing()

def get_center_from_mask(mask: np.ndarray):
    """
    """
    m = cv2.moments(mask) 
    return int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])

def test_warp_along_side():
    path = "/data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/results/exp010/0/data/smooth/0.png"

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.float32(face_parse(image))

    cv2.imwrite("face_mask.jpg", mask * 255)


    if mask.ndim < 3:
        mask = mask[..., np.newaxis]

    strength = -5
    h, w, c = image.shape
    erosion_size = 15
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    mask_erosion = cv2.erode(mask, element)
    mask_diff = mask - mask_erosion

    x,y = get_center_from_mask(mask_diff[...,0])
    x_linspace = np.linspace(0, w - 1, w)
    y_linspace = np.linspace(0, h - 1, h)
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

    offset_strength = (mask * strength).astype(np.float32)
    offset_strength = cv2.boxFilter(offset_strength, -1, ksize = (21, 21)) 
        
    offset_x = np.sign(x - x_grid) * offset_strength
    offset_y = np.sign(y - y_grid) * offset_strength
    warp_image = cv2.remap(image, (x_grid + offset_x).astype(np.float32), (y_grid + offset_y).astype(np.float32), cv2.INTER_LINEAR)
    warp_image = cv2.cvtColor(warp_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("xxx.jpg", warp_image)

    



