import os
import click

import torch
import argparse
import torch.multiprocessing as mp

from DeepLog import Timer

from .defs import aligner, kernel

@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--resume_path', default = None, help = "resume snapshots.")
@click.option('--gpus', default = 1)
def get_args(config_path,
             save_path,
             resume_path,
             gpus
            ):
    return config_path, save_path, resume_path, gpus


if __name__ == '__main__':
    t = Timer()
    os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    parse = argparse.ArgumentParser()
    parse.add_argument("--config_path")
    parse.add_argument("--save_path")
    parse.add_argument("--resume_path")
    parse.add_argument("--gpus", default = 1, type = int)
    config = parse.parse_args()

    gpus = config.gpus
    config_path = config.config_path
    save_path = config.save_path
    resume_path = config.resume_path
    t.tic("training start.")
    assert gpus >= 1, "expected gpu device more 1."
    if gpus <= 1:
        aligner(config_path, save_path, resume_path)
    else:
        world_size = gpus
        mp.spawn(
                 kernel,
                 args=(
                        world_size,
                        config_path,
                        save_path,
                        resume_path
                      ),
                 nprocs=world_size,
                 join=True
                )
    t.toc("training start.")
