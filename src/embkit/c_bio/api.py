import requests
import logging

logger = logging.getLogger(__name__)


class CBIOAPI:
    """
    Small client for the cBioPortal public API.

    cBioPortal is a public catalog of cancer genomics studies. This helper
    wraps the lightweight metadata endpoints (like listing available studies)
    that don't require downloading study data — use the
    :class:`~embkit.resources.c_bio_portal.CBIOPortal` resource for fetching an
    actual study's molecular profile data.
    """

    @staticmethod
    def list_studies():
        """
        List the cBioPortal studies available for download.

        Returns:
            The API's JSON list of studies, or ``None`` if the request failed.
        """
        try:
            studies = requests.get("https://www.cbioportal.org/api/studies").json()
            return studies
        except requests.RequestException as e:
            logger.error("Error fetching studies: %s", e)
            return None
