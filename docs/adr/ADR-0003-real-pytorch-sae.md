# ADR-0003: A real PyTorch TopK SAE, not a NumPy stand-in

- Status: Accepted
- Date: 2026-05-29

## Context

Sparse autoencoders are the core of white-box feature extraction. A common
shortcut is a linear projection dressed up as an SAE, which produces
non-monosemantic, untrustworthy features.

## Decision

`SparseAutoencoder` is implemented in real PyTorch with:

- exact `TopK(h, k)` structural sparsity,
- autograd training with Adam,
- decoder-column unit-norm renormalisation after every optimiser step (so the
  L1 penalty cannot be trivially defeated),
- dead-feature resampling,
- real diagnostics (reconstruction MSE, explained variance, L0 sparsity,
  dead-feature fraction).

## Consequences

- Requires the `huggingface` extra (torch) for training.
- `encode`/`decode` accept and return NumPy so use-sites stay torch-free.
- Decoder columns double as intervention directions (`feature_direction`).
