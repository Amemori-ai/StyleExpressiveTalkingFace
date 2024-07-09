import os
import sys
sys.path.insert(0, os.getcwd())

import click
import torch

@click.command()
@click.option("--path", default = None)
@click.option("--save_path", default = None)
def get_id(
           path,
           save_path
          ):
    
    _, _, _, selected_id = torch.load(path)

    with open(save_path, "w") as f:
        f.write(str(selected_id))

if __name__ == "__main__":
    get_id()


