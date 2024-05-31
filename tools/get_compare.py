"""get compare 
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import click
import tqdm

import cv2
import imageio
import numpy as np
import yaml
import json

from DeepLog import logger

                                

DEBUG = os.environ.get("DEBUG", False)
DEBUG = True if DEBUG in ["True", "TRUE", True, 1] else False

with open("/data1/wanghaoran/Amemori/template.yaml") as f:
    config = yaml.load(f, Loader = yaml.CLoader)

regions = eval(config["soft_mask_region"])
output_copy_region = eval(config["output_copy_region"])

def get_soft_mask_by_region():
    soft_mask  = np.zeros((512,512,3), np.float32)
    for region in regions:
        y1,y2,x1,x2 = region
        soft_mask[y1:y2,x1:x2,:]=1
    soft_mask = cv2.GaussianBlur(soft_mask, (101, 101), 11)
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

    output = output * partial_mask + master * (1 - partial_mask)
    if DEBUG:
        cv2.imwrite("merge_mask.jpg", merge_mask * 255.0)
        cv2.imwrite("partial_mask.jpg", partial_mask * 255.0)

    return np.clip(output, 0.0, 255.0).astype(np.uint8)

def get_compare(
                 gen_video_path: str,
                 gt_path: str,
                 to_path: str,
                 json_path: str
                 ) -> None:
    """get compare function.
    """
    reader = imageio.get_reader(gen_video_path)
    writer = imageio.get_writer(to_path, fps = 25)

    scores_all_frames = None
    threshold = 0.0
    if json_path is not None:
        with open(json_path) as f:
            scores_all_frames = json.load(f)
        threshold = 0.0185

    for index, image in enumerate(reader):
        if DEBUG:
            if index > 0:
                break
        if json_path is not None:
            if scores_all_frames[f'{index + 1}']['loss'] < threshold:
                logger.info("skip current frame.")
                continue
        image = image[:1024, :, ...]
        image = cv2.resize(image, (512,512))
        #gt_image = cv2.imread(os.path.join(gt_path, f'{index}_gen.jpg'))
        gt_image = cv2.imread(os.path.join(gt_path, f'{index}.png'))
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.resize(gt_image, (512,512))
        output = merge_from_two_image(gt_image, image)
        writer.append_data(output)
        

@click.command()
@click.option('--from_path')
@click.option('--gt_path')
@click.option('--to_path')
@click.option('--json_path', default = None)
def invoker(from_path,
            gt_path,
            to_path,
            json_path
            ):
    return get_compare(from_path, gt_path, to_path, json_path)

if __name__ == '__main__':
    invoker()

