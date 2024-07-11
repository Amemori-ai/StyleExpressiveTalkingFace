import click
from .aligner import aligner

@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--resume_path', default = None, help = "resume snapshots.")
def main(config_path,
         save_path,
         resume_path):

    return aligner(config_path, save_path, resume_path)

if __name__ == '__main__':
    main()
