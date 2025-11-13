# Object Detection Helpers

This package hosts the TensorFlow-style input pipelines, anchor generators, and loss functions required by the EfficientDet training loop.

## Contents
- Data parsers that translate TFRecord boundaries into the format consumed by the PyTorch conversion layer.
- Standard bounding-box coders and matcher utilities.
- Augmentation helpers for scale, shift, and color transformations.

## Implementation Notes
- Components are written to mirror the behavior of the original TensorFlow research implementation so that checkpoints remain interchangeable.
- When porting changes, prefer incremental verification to retain numerical parity with previously released models.