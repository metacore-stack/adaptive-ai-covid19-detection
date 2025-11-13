# Evaluation Toolkit

These modules provide the mean average precision, class-wise statistics, and summary writers used during validation of EfficientDet checkpoints.

## What Lives Here
- Dataset evaluators compatible with TFRecord inputs.
- Matching utilities that compute overlaps, assign true positives, and gather confusion metrics.
- Metric accumulation helpers that stream predictions across replicas before reporting aggregated values.

## Provenance
The implementation is derived from TensorFlow Object Detection evaluation utilities and remains under the Apache 2.0 license. Refer to the accompanying `LICENSE` file for the full legal text.

