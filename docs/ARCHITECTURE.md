# Architecture

## Overview

The model follows the NVIDIA DAVE-2 style approach for end-to-end steering-angle prediction from RGB road images. Input frames are cropped, converted to YUV, resized to `66 x 200`, normalized to `[0, 1]`, and then passed through a compact convolutional regression network.

## Model Pipeline

```text
Raw RGB image
  -> crop road region
  -> convert RGB to YUV
  -> resize to 66 x 200
  -> normalize to [0, 1]
  -> CNN regression model
  -> steering angle
```

## Network Layout

```text
Input: (66, 200, 3)
Conv2D 24 filters, 5x5, stride 2, ELU
Conv2D 36 filters, 5x5, stride 2, ELU
Conv2D 48 filters, 5x5, stride 2, ELU
Conv2D 64 filters, 3x3, ELU
Conv2D 64 filters, 3x3, ELU
Flatten
Dense 100, ELU
Dense 50, ELU
Dense 10, ELU
Dense 1
```

## Parameter Count

The current implementation contains `252,219` trainable parameters.

## Data Augmentation

Training batches use stochastic augmentation to improve generalization:

- random camera selection across center, left, and right views
- steering-angle offsets for side cameras
- random translation with steering compensation
- random brightness reduction
- random horizontal flipping with sign inversion

Validation batches skip augmentation but still use the same preprocessing pipeline.

## Training Notes

The default training CLI is intentionally conservative so it can be tuned per dataset:

- batch size: `64`
- epochs: `5`
- steps per epoch: `200`
- validation steps: `50`
- optimizer: `Adam`
- learning rate: `1e-4`
- loss: `MSE`
- metric: `MAE`

These values are safe starting points, not fixed benchmark settings.

## Operational Considerations

- The repository expects a Udacity-style `driving_log.csv`.
- Camera paths are normalized relative to the CSV location when possible.
- Realtime inference is served over Socket.IO for compatibility with the simulator bridge.

## References

- NVIDIA, *End to End Learning for Self-Driving Cars*
- Udacity Self-Driving Car Simulator
- TensorFlow Keras documentation
