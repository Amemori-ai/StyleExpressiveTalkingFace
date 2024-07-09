import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from collections import OrderedDict

def swap_model(
                src_model_path,
                ref_model_path,
                dest_model_path
              ):

    src_state_dict = torch.load(src_model_path)["weight"]
    ref_state_dict = torch.load(ref_model_path)["weight"]
    dest_state_dict = OrderedDict()

    for k, v in src_state_dict.items():
        if "_max" in k or "_min" in k or "clip_values" in k:
            dest_state_dict[k] = ref_state_dict[k]
        else:
            dest_state_dict[k] = v

    torch.save(dest_state_dict, dest_model_path)


if __name__ == "__main__":

    src = sys.argv[1]
    ref = sys.argv[2]
    dest = sys.argv[3]
    dest = os.path.join(dest, "snapshots")
    os.makedirs(dest, exist_ok = True)

    dest_path = os.path.join(dest, "best.pth")
    swap_model(src, ref, dest_path)


