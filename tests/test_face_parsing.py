import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import numpy as np
from TalkingFace.aligner import face_parsing

def test_face_parsing():
    face_parse = face_parsing()
    path = "/data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding/results/exp010/0/data/smooth/0.png"
    image = cv2.imread(path)[...,::-1]
    mask = face_parse(image)
    cv2.imwrite("mask_total.jpg", mask * 255.0)
