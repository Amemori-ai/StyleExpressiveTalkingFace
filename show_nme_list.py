import os
import torch
from DeepLog import logger

def get_range(
              _range: str
             ):
    start, end = _range.split(":")
    return int(start), int(end)

def get_exp_nme(
                 _range: str,
                 default_model_path: str = "./results/exp0{}/snapshots/best.pth"
                 ):
    
    nme_show = {}
    start, end = get_range(_range)
    for idx in range(start, end + 1):
        _model_path = default_model_path.format(idx)
        nme = torch.load(_model_path)['best_value']
        nme_show[f"exp0{idx}"] = nme
    logger.info(nme_show)


if __name__ == "__main__":
    import sys
    exp_range = sys.argv[1]
    get_exp_nme(exp_range)


