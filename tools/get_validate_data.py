"""get validate data.
"""
from typing import List
import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import click

import numpy as np

from functools import singledispatch

def load(path):
    if path.endswith('pt'):
        return torch.load(path)
    elif path.endswith('npy'):
        return np.load(path)

@singledispatch
def save(path, 
         obj):
    pass

@save.register
def _(
        obj: list,
        path: str
     ):
    
    torch.save(obj, path + '_attr.pt')

@save.register
def _(
        obj: np.ndarray,
        path: str
     ):
    np.save(path + '_landmarks.npy', obj)

def get_validate_data(
                        from_path: str,
                        ratio: float,
                        to_path: str
                     ):
 
    _obj = load(from_path)
    length = round(len(_obj) * ratio)

    # validate and train path.
    train_path = os.path.join(to_path, "train")
    val_path = os.path.join(to_path, "val")
    
    save(_obj[:length], train_path)
    save(_obj[length:], val_path)


@click.command()
@click.option('--from_path')
@click.option('--to_path')
@click.option('--ratio', type = click.FLOAT)
def invoker(
            from_path: str,
            to_path: str,
            ratio: float
           ):

    return get_validate_data(from_path,
                             ratio,
                             to_path)


if __name__ == '__main__':
    invoker()

