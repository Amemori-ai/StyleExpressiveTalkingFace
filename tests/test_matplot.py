import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt

from TalkingFace.aligner import _minmax_constant_no_clip

def draw(
         x: np.ndarray,
         path: str
         ):
        

    fig, ax = plt.subplots()
    #k, c = x.shape # keypoints, channels

    #x_axis = np.linspace(1, k, k)

    #ax.bar(x_axis, x.reshape(-1))
    ax.hist(x.reshape(-1), [-2,-1,0,1,2], (-2, 2))
    ax.set_ylabel("xxx")
    ax.set_title("key points distributioin.")
    plt.savefig(path)

def get_min_max(path:str):
    state_dict = torch.load(path)["weight"]
        
    def to_numpy(x):
        return x.detach().cpu().numpy()

    return to_numpy(state_dict["_min"]), to_numpy(state_dict["_max"])

def test_draw_plot_from_numpy_array():

    array_path = "/data1/wanghaoran/Amemori/StyleExpressiveTalkingFace/tests/offset.npy"
    numpy_array = np.load(array_path)

    min_max_path = "/data1/wanghaoran/Amemori/StyleExpressiveTalkingFace/results/exp103/snapshots/best.pth"
    _min, _max = get_min_max(min_max_path)

    numpy_array = _minmax_constant_no_clip(numpy_array, _min, _max)

    save_path = "/data1/wanghaoran/Amemori/StyleExpressiveTalkingFace/tests/fig.jpg"
    draw(numpy_array, save_path)

    #for _array in numpy_array:
        
        





        



