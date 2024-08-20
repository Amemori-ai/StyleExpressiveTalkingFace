import os
import torch
import click
import numpy as np 

def get_id_landmark(    
                    id_path,
                    landmark_path,
                    to_path
                    ):

    gen_files_list, selected_id_image, selected_id_latent, selected_id = torch.load(id_path)
    landmarks = np.load(landmark_path)
    # np.save(to_path, landmarks[selected_id - 1: selected_id, ...])
    print(selected_id)
    np.save(to_path, landmarks[selected_id: selected_id+1, ...])

@click.command()
@click.option('--id_path')
@click.option('--landmark_path')
@click.option('--to_path')
def _invoker_func(
                    id_path,
                    landmark_path,
                    to_path
                    ):
    return get_id_landmark(id_path, landmark_path, to_path)

if __name__ == '__main__':
    _invoker_func()


