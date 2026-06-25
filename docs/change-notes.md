# Change Notes (Unstaged Refactor)

This page summarizes the current unstaged refactor across model verification, masked-layer constraint enforcement, and training/serialization behavior.

## Scope

The unstaged changes touch:

- `src/embkit/commands/model.py`
- `src/embkit/factory/core.py`
- `src/embkit/models/vae/base_vae.py`
- `src/embkit/models/vae/net_vae.py`
- `src/embkit/models/vae/rna_vae.py`
- `src/embkit/models/vae/vae.py`
- `src/embkit/models/ffnn.py`
- `src/embkit/modules/masked_linear.py`
- `src/embkit/optimize/__init__.py`
- tests for command and verification behavior

## Verification Command Updates

`embkit model verify` now supports:

- `--json` for machine-readable output
- `--ci` for CI-safe behavior (`--json` + fail on unhealthy)
- `--fail-on-unhealthy`
- strict identity checks:
  - `--strict`
  - `--expected-feature-count`
  - `--expected-latent-dim`
  - `--expected-features-file`

The command wording was updated from “authenticity/paranoid” framing to “integrity/sanity” framing.

## Why Mask Clamping Was Added

Pathway masking is now enforced at three points:

1. Forward pass: use effective masked weights (`weight * mask`).
2. Post-optimizer step: clamp masked entries back to zero.
3. Save-time safety net: clamp again before serialization.

Reason:

- Forward masking alone guarantees functional behavior, but raw masked parameters can still drift from zero due to optimizer state.
- Post-step clamping preserves parameter-level invariants and prevents leakage in saved checkpoints.
- Save-time clamping guarantees serialized artifacts remain constraint-consistent.

## Training and Serialization Behavior

- Optimizer loops now enforce mask clamping after every step.
- `factory.save(...)` clamps constrained weights before writing checkpoints.
- Model `history` fields are normalized to dictionaries for stable serialization/deserialization (`{}` fallback instead of `None`).

## Verification Report Metadata

Verification now includes identity metadata when available:

- `feature_names`
- `features_count`
- `declared_latent_dim`

This enables strict identity checks from CLI and CI workflows.

## NetVAE Leakage Audit Adjustment

NetVAE verification snapshots constrained layer weights before deep audit forwards run. This prevents false negatives where a later forward pass could zero leaked raw entries before the audit reads them.

## Expected Outcome

- Models trained and saved with this refactor should no longer show large masked-edge leakage in verification reports.
- Verification output is more actionable for both interactive usage and CI enforcement.
