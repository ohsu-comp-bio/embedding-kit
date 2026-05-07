import unittest
import torch
import os
import tempfile
import numpy as np
from embkit.models.vae.net_vae import NetVAE
from embkit.models.vae.rna_vae import RNAVAE
from embkit.models.ffnn import FFNN
from embkit.factory import save, run_model_verification

class TestModelVerification(unittest.TestCase):
    def setUp(self):
        self.features = ["G1", "G2", "G3", "G4"]
        self.latent_groups = {"P1": ["G1", "G2"], "P2": ["G3", "G4"]}
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_net_vae_verification_healthy(self):
        model = NetVAE(features=self.features, latent_groups=self.latent_groups)
        model.history = {"loss": [10.0, 5.0, 2.0]}
        path = os.path.join(self.temp_dir.name, "netvae.model")
        save(model, path)
        
        report = run_model_verification(path)
        self.assertTrue(report["healthy"])
        self.assertEqual(report["model_type"], "NetVAE")
        self.assertIn("history_summary", report)

    def test_net_vae_leakage_failure(self):
        model = NetVAE(features=self.features, latent_groups=self.latent_groups)
        model.history = {"loss": [10.0, 5.0, 2.0]}
        # Poison a weight outside the mask
        with torch.no_grad():
            module = model.encoder.net[0]
            mask = module.mask
            # Find a zero in the mask
            idx = (mask == 0).nonzero(as_tuple=True)
            if len(idx[0]) > 0:
                # Add a significant weight where it should be zero
                module.linear.weight[idx[0][0], idx[1][0]] = 1.0

        # In-memory verifier should detect leakage.
        in_memory_report = model.verify_integrity()
        self.assertFalse(in_memory_report["healthy"])
        self.assertTrue(any("leakage" in i.lower() for i in in_memory_report["issues"]))

        # Serialization safety net clamps masked weights before save.
        path = os.path.join(self.temp_dir.name, "leakage.model")
        save(model, path)
        report = run_model_verification(path)
        self.assertTrue(report["healthy"])
        self.assertFalse(any("leakage" in i.lower() for i in report["issues"]))

    def test_rna_vae_verification(self):
        model = RNAVAE(features=self.features, latent_dim=2)
        model.history = {"loss": [1.0, 0.5]}
        path = os.path.join(self.temp_dir.name, "rnavae.model")
        save(model, path)
        
        report = run_model_verification(path)
        self.assertTrue(report["healthy"])
        self.assertIn("rna_diagnostics", report)
        self.assertTrue(report["rna_diagnostics"]["is_non_negative"])

    def test_ffnn_history_failure(self):
        model = FFNN(input_dim=4, output_dim=1)
        # Loss went UP - failed to improve
        model.history = {"loss": [0.1, 0.5]} 
        path = os.path.join(self.temp_dir.name, "bad_history.model")
        save(model, path)
        
        report = run_model_verification(path)
        self.assertFalse(report["healthy"])
        self.assertTrue(any("failed to improve" in i for i in report["issues"]))

    def test_corrupt_nan_verification(self):
        model = FFNN(input_dim=4, output_dim=1)
        with torch.no_grad():
            params = list(model.parameters())
            params[0][0] = float('nan')
        
        path = os.path.join(self.temp_dir.name, "nan.model")
        save(model, path)
        
        report = run_model_verification(path)
        self.assertFalse(report["healthy"])
        self.assertTrue(any("NaN" in issue for issue in report["issues"]))

if __name__ == "__main__":
    unittest.main()
