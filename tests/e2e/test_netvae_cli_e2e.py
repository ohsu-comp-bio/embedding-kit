import unittest
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from click.testing import CliRunner

from embkit.__main__ import cli_main
from embkit.factory import load
from embkit.losses import bce_with_logits
from embkit.modules import MaskedLinear
from embkit.pathway import (
    build_features_to_group_mask,
    extract_sif_interactions,
    feature_map_intersect,
)

logger = logging.getLogger("tests.e2e.netvae_cli")


class TestNetVAECLIE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def setUp(self):
        self.runner = CliRunner()
        self.data_dir = Path(__file__).resolve().parent / "data"
        self.expr_tsv = self.data_dir / "toy_expr.tsv"
        self.pathway_sif = self.data_dir / "toy_pathway.sif"

    def test_train_and_encode_netvae_with_constraints(self):
        with self.runner.isolated_filesystem():
            model_path = "toy_netvae.model"
            embed_path = "toy_embedding.tsv"
            print("[E2E] Starting NetVAE train on toy data")
            logger.info("E2E start: training toy NetVAE model")

            train_result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-netvae",
                    str(self.expr_tsv),
                    str(self.pathway_sif),
                    "--epochs",
                    "2",
                    "--group-layer-size",
                    "1",
                    "--save-stats",
                    "--out",
                    model_path,
                ],
            )
            self.assertEqual(train_result.exit_code, 0, msg=train_result.output)
            print("[E2E] Train command finished")
            logger.info("Train command completed. Output:\n%s", train_result.output.strip())
            self.assertTrue(Path(model_path).exists())
            self.assertTrue(Path(f"{model_path}.stats.tsv").exists())
            logger.info("Artifacts created: %s and %s.stats.tsv", model_path, model_path)

            logger.info("Running encode command against trained model")
            print("[E2E] Running encode command")
            encode_result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "encode",
                    str(self.expr_tsv),
                    model_path,
                    "--out",
                    embed_path,
                ],
            )
            self.assertEqual(encode_result.exit_code, 0, msg=encode_result.output)
            print("[E2E] Encode command finished")
            logger.info("Encode command completed. Output:\n%s", encode_result.output.strip())
            self.assertTrue(Path(embed_path).exists())

            embedding = pd.read_csv(embed_path, sep="\t", index_col=0)
            self.assertEqual(embedding.shape, (10, 3))
            print("[E2E] Embedding shape validated:", embedding.shape)
            logger.info("Embedding shape validated: %s", embedding.shape)

            model = load(model_path, device=torch.device("cpu"))
            self.assertIsNotNone(model.encoder)
            self.assertIsNotNone(model.decoder)
            logger.info("Model loaded and encoder/decoder present")

            first_masked = next(m for m in model.encoder.net if isinstance(m, MaskedLinear))
            self.assertEqual(tuple(first_masked.mask.shape), (3, 10))
            print("[E2E] Masked layer shape validated:", tuple(first_masked.mask.shape))
            logger.info("First masked encoder layer shape: %s", tuple(first_masked.mask.shape))

            expr_df = pd.read_csv(self.expr_tsv, sep="\t", index_col=0)
            fmap = extract_sif_interactions(str(self.pathway_sif))
            fmap = feature_map_intersect(fmap, expr_df.columns)
            expected_mask = build_features_to_group_mask(
                fmap,
                feature_idx=model.features,
                group_idx=model.latent_index,
                group_node_count=1,
            )

            np.testing.assert_array_equal(first_masked.mask.detach().cpu().numpy(), expected_mask)
            print("[E2E] Constraint mask equality check passed")
            logger.info(
                "Mask equality check passed. allowed_edges=%d blocked_edges=%d",
                int(expected_mask.sum()),
                int(expected_mask.size - expected_mask.sum()),
            )

            constraint_info = getattr(first_masked, "constraint_info", None)
            self.assertIsNotNone(constraint_info)
            self.assertTrue(getattr(constraint_info, "active", False))

            effective_weight = first_masked.linear.weight.detach() * first_masked.mask.detach()
            self.assertTrue(torch.all(effective_weight[first_masked.mask == 0] == 0))
            self.assertTrue(torch.any(torch.abs(effective_weight[first_masked.mask == 1]) > 0))
            print("[E2E] Effective masked weights validated")
            logger.info("Effective masked weights validated (blocked=0, allowed has signal)")

            # Explicitly verify constrained edges are blocked during optimization.
            x = torch.tensor(expr_df[model.features].values, dtype=torch.float32)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            opt.zero_grad()
            mu, logvar, z = model.encoder(x)
            recon = model.decoder(z)
            total, _, _ = bce_with_logits(recon, x, mu, logvar, beta=1.0)
            total.backward()

            weight = first_masked.linear.weight
            mask = first_masked.mask
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.all(weight.grad[mask == 0] == 0))
            print("[E2E] Gradient blocking validated")
            logger.info("Gradient blocking validated on constrained edges")

            blocked_before = weight.detach()[mask == 0].clone()
            opt.step()
            blocked_after = weight.detach()[mask == 0]
            self.assertTrue(torch.all(blocked_after == blocked_before))
            print("[E2E] No-update invariant validated on constrained parameters")
            logger.info("No-update invariant validated for constrained raw parameters")
            logger.info("E2E NetVAE constraint test finished successfully")


if __name__ == "__main__":
    unittest.main()
