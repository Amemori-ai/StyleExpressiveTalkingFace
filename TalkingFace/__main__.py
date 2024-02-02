import click
from .aligner import aligner

@click.command()
@click.option('--config_path')
@click.option('--save_path')
def main(config_path,
         save_path):
    return aligner(config_path, save_path)

if __name__ == '__main__':
    main()
