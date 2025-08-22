import click
from ..datasets import CBIOPortal
from ..c_bio import CBIOAPI
import logging

logger = logging.getLogger(__name__)

cbio_cmd = click.Group(name="cbio", help="cBIO commands.")


@cbio_cmd.command(name="studies", help="List cbio studies.")
@click.pass_context
def list_studies(ctx):
    """
    Command to list available cBIO studies.

    Parameters:
        ctx: Click context object.

    Returns:
        None
    """
    api = CBIOAPI()
    studies = api.list_studies()

    if studies:
        for study in studies:
            click.echo(f"Study ID: {study['studyId']}, Name: {study['name']}")
    else:
        click.echo("No studies found or an error occurred while fetching studies.")


@cbio_cmd.command(name="download", help="Download cbio dataset.")
@click.option("--save_path", "-s", type=click.Path(exists=False, writable=True, path_type=str),
              help="Path to save the dataset.")
@click.option("--study_id", "-si", type=str, show_default=True, help="Id of the study to download.")
@click.pass_context
def download_tmp(ctx, save_path: str | None, study_id: str):
    """
    Command to download a cBIO dataset by specifying the study ID.


    Parameters:
        ctx: Click context object.
        save_path: Path to save the dataset.
        study_id: ID of the study to download.

    Returns:
        None
    """

    if study_id is None:
        click.echo("Study Id not specified. Please use -si option to specify the study id.")
        return

    cbio_portal: CBIOPortal = CBIOPortal(save_path=save_path, study_id=study_id, download=True)
    cbio_portal.download()
    cbio_portal.unpack()


# needed for testing
if __name__ == "__main__":
    cbio_cmd()  # pragma: no cover
