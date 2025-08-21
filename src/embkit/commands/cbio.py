import click
from ..datasets import CBIOPortal
from ..c_bio import CBIOAPI
import logging

logger = logging.getLogger(__name__)

cbio = click.Group(name="cbio", help="cBIO commands.")

@cbio.command(name="studies", help="List cbio studies.")
@click.pass_context
def list_studies(ctx):
    """List all available cBIO studies."""
    api = CBIOAPI()
    studies = api.list_studies()

    if studies:
        for study in studies:
            click.echo(f"Study ID: {study['studyId']}, Name: {study['name']}")
    else:
        click.echo("No studies found or an error occurred while fetching studies.")


@cbio.command(name="download", help="Download cbio dataset.")
@click.option("--save_path", "-s", type=click.Path(exists=False, writable=True, path_type=str), help="Path to save the dataset.")
@click.option("--study_name", "-sn", type=str,  show_default=True, help="Name of the study to download.")
@click.pass_context
def download_tmp(ctx, save_path: str | None, study_name: str):
    """Download the TMP dataset."""

    if study_name is None:
        click.echo("Study name not specified. Please use --s option to specify the study name.")
        return

    cbio_portal: CBIOPortal = CBIOPortal(save_path=save_path, study_name=study_name, download=True)
    cbio_portal.download()
    cbio_portal.unpack()


# needed for testing
if __name__ == "__main__":
    cbio() # pragma: no cover