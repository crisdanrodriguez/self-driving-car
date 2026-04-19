"""Training entry point for the steering-angle prediction model."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_STEPS_PER_EPOCH = 200
DEFAULT_VALIDATION_STEPS = 50
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_RANDOM_STATE = 42


def train_model(
    data_path: str = "data/driving_log.csv",
    model_output: str = "model.h5",
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH,
    validation_steps: int = DEFAULT_VALIDATION_STEPS,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[object, object]:
    """Train the CNN model using a Udacity-style driving log."""

    from sklearn.model_selection import train_test_split

    from self_driving_car.model import nvidia_model
    from self_driving_car.preprocessing import batch_generator, prepare_driving_log

    driving_log = prepare_driving_log(data_path)

    train_df, val_df = train_test_split(
        driving_log,
        test_size=validation_split,
        random_state=random_state,
        shuffle=True,
    )

    print(f"Total samples imported: {len(driving_log)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    model = nvidia_model()
    history = model.fit(
        batch_generator(train_df, batch_size=batch_size, training=True),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=batch_generator(val_df, batch_size=batch_size, training=False),
        validation_steps=validation_steps,
        verbose=1,
    )

    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")

    return model, history


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for training."""

    parser = argparse.ArgumentParser(
        description="Train the NVIDIA DAVE-2 style model on a driving_log.csv dataset.",
    )
    parser.add_argument(
        "--data-path",
        default="data/driving_log.csv",
        help="Path to the driving_log.csv file.",
    )
    parser.add_argument(
        "--model-output",
        default="model.h5",
        help="Destination path for the trained model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=DEFAULT_STEPS_PER_EPOCH,
        help="Batches to process per epoch.",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=DEFAULT_VALIDATION_STEPS,
        help="Validation batches to process after each epoch.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for train/validation splitting.",
    )
    return parser


def main() -> None:
    """Run model training from the command line."""

    args = build_parser().parse_args()

    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.steps_per_epoch <= 0 or args.validation_steps <= 0:
        raise ValueError("Training and validation steps must be greater than 0.")

    train_model(
        data_path=args.data_path,
        model_output=args.model_output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        validation_split=args.validation_split,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
