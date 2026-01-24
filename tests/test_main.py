import subprocess
import sys
import unittest
from click.testing import CliRunner
from embkit.__main__ import cli_main


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_no_args_shows_help(self):
        result = self.runner.invoke(cli_main, [])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Embedding Kit CLI", result.output)
        self.assertIn("Commands:", result.output)

    def test_help_command(self):
        result = self.runner.invoke(cli_main, ["help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Embedding Kit CLI", result.output)
        self.assertIn("Commands:", result.output)

    def test_help_model(self):
        result = self.runner.invoke(cli_main, ["help", "model"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.output)

    def test_help_matrix(self):
        result = self.runner.invoke(cli_main, ["help", "matrix"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.output)

    def test_unknown_command(self):
        result = self.runner.invoke(cli_main, ["help", "nonsense"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown command", result.output)

    def test_nested_invalid_command(self):
        result = self.runner.invoke(cli_main, ["help", "model", "invalid"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown command", result.output)

    def test_main_entrypoint(self):
        """Test `python -m embkit` entrypoint directly."""
        result = subprocess.run(
            [sys.executable, "-m", "embkit"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Embedding Kit CLI", result.stdout)

    def test_help_command_direct_via_cli_group(self):
        """Force invoking the top-level CLI help manually as 'embkit help' does on command line."""
        result = self.runner.invoke(cli_main, ['help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Embedding Kit CLI", result.output)
        self.assertIn("Commands:", result.output)
        # Force-triggers line 52: ctx.parent.get_help()
        self.assertTrue(result.output.count("model") > 0 or result.output.count("matrix") > 0)

    def test_version_flag(self):
        """Test that --version flag shows version information."""
        result = self.runner.invoke(cli_main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("embedding-kit version", result.output)
        # Should include version number from package metadata
        self.assertRegex(result.output, r'embedding-kit version \d+\.\d+')

    def test_version_flag_with_git_info(self):
        """Test that --version includes git metadata when available."""
        result = self.runner.invoke(cli_main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        # When running in a git repo, should include commit, branch, and remote
        if "commit:" in result.output:
            self.assertIn("commit:", result.output)
            self.assertIn("branch:", result.output)
            self.assertIn("remote:", result.output)

    def test_version_entrypoint(self):
        """Test `python -m embkit --version` entrypoint."""
        result = subprocess.run(
            [sys.executable, "-m", "embkit", "--version"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("embedding-kit version", result.stdout)
        self.assertRegex(result.stdout, r'embedding-kit version \d+\.\d+')


if __name__ == '__main__':
    unittest.main()