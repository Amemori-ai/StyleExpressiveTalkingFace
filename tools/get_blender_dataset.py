import os
import sys
sys.path.insert(0, os.getcwd())
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from TalkingFace.aligner import face_parsing
from TalkingFace.util import make_dataset

face_parse = face_parsing()

def get_diff_mask(
                    mask
                  ):
    
    
     erosion_size = 20
     element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
     mask_erosion = cv2.erode(mask, element)
     dilate_size = 5
     element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_size + 1, 2 * dilate_size + 1), (dilate_size, dilate_size))
     mask_dilate = cv2.dilate(mask, element)
     return np.uint8(mask_dilate - mask_erosion)

def get_training_dataset(
                         driving_path: str,
                         save_path: str
                         ):

    gt_image_path = os.path.join(save_path, "gt")
    mask_image_path = os.path.join(save_path, "mask")

    os.makedirs(gt_image_path, exist_ok = True)
    os.makedirs(mask_image_path, exist_ok = True)

    driving_files = sorted(make_dataset(driving_path), \
                        key = lambda x: int(os.path.basename(x[1]).split('.')[0]))

    pbar = tqdm(driving_files)
    for _file in pbar:
        _file = _file[1]
        _file_name = os.path.basename(_file)
        image = cv2.imread(_file)[...,::-1]
        mask = face_parse(image).astype(np.float32)
        mask_diff = get_diff_mask(mask) * 255.0
        cv2.imwrite(os.path.join(mask_image_path, _file_name), mask_diff)
        shutil.copy(_file, os.path.join(gt_image_path, _file_name))



    
if __name__ == "__main__":

    driving_path = sys.argv[1]
    save_path = sys.argv[2]
    get_training_dataset(driving_path, save_path)
