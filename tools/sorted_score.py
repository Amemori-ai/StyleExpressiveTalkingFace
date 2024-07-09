"""get sorted scores.
"""
import os
import sys

import json
import click
import shutil
from collections import OrderedDict

def get_sorted_score_frame(
                            from_path :str,
                            json_path : str,
                            to_path: str
                          ):
    """get sorted scores frame
    """

    os.makedirs(to_path, exist_ok = True)
    with open(json_path) as f:
        scores = json.load(f)
    scores = dict(sorted(scores.items(), key = lambda x: x[1]['loss']))

    for k, v in scores.items():
        if v['loss'] > 0.0185:
            shutil.copy(os.path.join(from_path, f'{k}.jpg'), os.path.join(to_path, f'{k}.jpg'))


@click.command()
@click.option('--from_path')
@click.option('--json_path')
@click.option('--to_path')
def _invoker(from_path,
             json_path,
             to_path
             ):
    return get_sorted_score_frame(from_path, json_path, to_path)

if __name__ == '__main__':
    _invoker()
