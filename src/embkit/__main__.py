
  # my_project/cli.py
import click
from .commands import model

@click.group()
def cli():
    """A multi-module CLI application."""
    pass

cli.add_command(model)

if __name__ == '__main__':
    cli()