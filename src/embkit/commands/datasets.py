import click
import pandas as pd
from ..datasets import GTEx

datasets = click.Group(name="datasets", help="Datasets commands.")


@datasets.command()
@click.option("--data_type", "-t", required=False, type=str, help="Name of the dataset to download.")
@click.option("--output_folder", "-f", required=True, type=str, help="The folder to download the dataset into.")
def gtex(data_type: str, output_folder: str):
    """Download GTEx dataset."""

    if data_type not in GTEx.NAMES:
        click.echo(f"Dataset name '{data_type}' is not recognized. Available options are: {list(GTEx.NAMES.keys())}")
        return

    GTEx(data_type=data_type, save_path=output_folder)

