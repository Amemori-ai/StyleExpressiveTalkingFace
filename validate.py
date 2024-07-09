import click
from TalkingFace.aligner import evaluate

@click.command()
@click.option("--config_path", default = None)
@click.option("--offset_weight_path", default = None)
@click.option("--pti_weight_path", default = None)
def invoker(
             config_path,
             offset_weight_path,
             pti_weight_path
           ):
    return evaluate(config_path, offset_weight_path, pti_weight_path)

if __name__ == "__main__":
    invoker()
       
