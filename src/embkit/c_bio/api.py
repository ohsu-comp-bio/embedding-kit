import requests
import logging

logger = logging.getLogger(__name__)


class CBIOAPI:

    @staticmethod
    def list_studies():
        try:
            studies = requests.get("https://www.cbioportal.org/api/studies").json()
            return studies
        except requests.RequestException as e:
            logger.error("Error fetching studies: %s", e)
            return None
