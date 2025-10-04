# ALPINE Strict Dataset - No Self-Loops

This dataset is derived from `composition_90_alpine_strict` with all self-loops removed.

## Removed patterns:
- S1 → S1 (self-loops within S1)
- S2 → S2 (self-loops within S2)
- S3 → S3 (self-loops within S3)

## Kept patterns:
- S1 → S2 (forward edges)
- S2 → S3 (forward edges)
- S1 → S3 (skip connections)

This modification allows cleaner analysis of weight gaps between different stage transitions.
