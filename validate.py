import click
import numpy as np
from TalkingFace.aligner import evaluate

@click.command()
@click.option("--config_path", default = None)
@click.option("--offset_weight_path", default = None)
@click.option("--pti_weight_path", default = None)
@click.option("--res_path", default = None)
def invoker(
             config_path,
             offset_weight_path,
             pti_weight_path,
             res_path
           ):
    res = evaluate(config_path, offset_weight_path, pti_weight_path)
    np.save(res_path, res)

if __name__ == "__main__":
    invoker()
       
