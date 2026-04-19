"""CNN model definition for steering-angle prediction."""

from __future__ import annotations

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam


def nvidia_model(input_shape: tuple[int, int, int] = (66, 200, 3)) -> Sequential:
    """Create and compile the NVIDIA DAVE-2 inspired CNN model."""

    model = Sequential(
        [
            Input(shape=input_shape, name="camera_frame"),
            Conv2D(24, (5, 5), strides=(2, 2), activation="elu", name="conv1"),
            Conv2D(36, (5, 5), strides=(2, 2), activation="elu", name="conv2"),
            Conv2D(48, (5, 5), strides=(2, 2), activation="elu", name="conv3"),
            Conv2D(64, (3, 3), activation="elu", name="conv4"),
            Conv2D(64, (3, 3), activation="elu", name="conv5"),
            Flatten(name="flatten"),
            Dense(100, activation="elu", name="dense1"),
            Dense(50, activation="elu", name="dense2"),
            Dense(10, activation="elu", name="dense3"),
            Dense(1, name="steering_angle"),
        ],
        name="nvidia_dave2",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main() -> None:
    """Print a short model summary when the module is executed directly."""

    model = nvidia_model()
    model.summary()
    print(f"Total parameters: {model.count_params():,}")


if __name__ == "__main__":
    main()
