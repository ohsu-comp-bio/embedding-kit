import requests


class CBIOAPI:

    @staticmethod
    def list_studies():
        try:
            studies = requests.get("https://www.cbioportal.org/api/studies").json()
            return studies
        except requests.RequestException as e:
            print(f"Error fetching studies: {e}")
            return None
