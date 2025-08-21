import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from embkit.commands.cbio import cbio_cmd


class TestCBIOCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("embkit.commands.cbio.CBIOAPI")
    def test_list_studies_displays_results(self, mock_cbioapi_cls):
        mock_api = MagicMock()
        mock_api.list_studies.return_value = [
            {"studyId": "study1", "name": "Study One"},
            {"studyId": "study2", "name": "Study Two"},
        ]
        mock_cbioapi_cls.return_value = mock_api

        result = self.runner.invoke(cbio_cmd, ["studies"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Study ID: study1", result.output)
        self.assertIn("Study ID: study2", result.output)

    @patch("embkit.commands.cbio.CBIOAPI")
    def test_list_studies_no_results(self, mock_cbioapi_cls):
        mock_api = MagicMock()
        mock_api.list_studies.return_value = []
        mock_cbioapi_cls.return_value = mock_api

        result = self.runner.invoke(cbio_cmd, ["studies"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("No studies found", result.output)

    @patch("embkit.commands.cbio.CBIOPortal")
    def test_download_command_runs_download_and_unpack(self, mock_cbioportal_cls):
        mock_portal = MagicMock()
        mock_cbioportal_cls.return_value = mock_portal

        result = self.runner.invoke(
            cbio_cmd,
            ["download", "--study_id", "studyX", "--save_path", "/tmp/testcbio"]
        )

        self.assertEqual(result.exit_code, 0)
        mock_cbioportal_cls.assert_called_once_with(
            save_path="/tmp/testcbio", study_id="studyX", download=True
        )
        mock_portal.download.assert_called_once()
        mock_portal.unpack.assert_called_once()

    def test_download_missing_study_id(self):
        result = self.runner.invoke(cbio_cmd, ["download"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Study Id not specified", result.output)


if __name__ == "__main__":
    unittest.main()