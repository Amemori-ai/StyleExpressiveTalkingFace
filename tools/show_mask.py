import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import numpy as np


def get_mask(
              from_path,
              save_path
            ):

    os.makedirs(save_path, exist_ok = True)

    lists = np.load(from_path)

    for i, a in enumerate(lists):
        cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), a * 255)


if __name__ == "__main__":

    from_path = sys.argv[1]
    to_path = sys.argv[2]
    get_mask(from_path, to_path)
