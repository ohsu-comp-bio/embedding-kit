import unittest
from unittest.mock import patch, MagicMock
from embkit.c_bio import CBIOAPI  # Adjust import path if different
import requests


class TestCBIOAPI(unittest.TestCase):

    @patch("embkit.c_bio.api.requests.get")
    def test_list_studies_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [{"studyId": "brca_tcga", "name": "Breast Cancer"}]
        mock_get.return_value = mock_response

        result = CBIOAPI.list_studies()
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["studyId"], "brca_tcga")

    @patch("embkit.c_bio.api.requests.get")
    def test_list_studies_request_exception(self, mock_get):
        mock_get.side_effect = requests.RequestException("Network error")

        result = CBIOAPI.list_studies()
        self.assertIsNone(result)

    @patch("embkit.c_bio.api.requests.get")
    def test_list_studies_json_exception(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        with self.assertRaises(ValueError):
            CBIOAPI.list_studies()


if __name__ == "__main__":
    unittest.main()