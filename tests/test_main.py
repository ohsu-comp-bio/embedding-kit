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


if __name__ == '__main__':
    unittest.main()