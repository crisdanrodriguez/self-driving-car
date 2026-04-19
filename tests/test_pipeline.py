"""Basic test suite for repository health and ML pipeline utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from self_driving_car.preprocessing import (
    batch_generator,
    brightness_reduction,
    horizontal_flip,
    image_preprocessing,
    prepare_driving_log,
    top_bottom_crop,
    translation,
)


def _tensorflow():
    return pytest.importorskip("tensorflow")


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(160, 320, 3), dtype=np.uint8)


def test_horizontal_flip_negates_steering(sample_image: np.ndarray) -> None:
    flipped_image, flipped_angle = horizontal_flip(sample_image.copy(), 0.35)

    assert flipped_image.shape == sample_image.shape
    assert flipped_angle == pytest.approx(-0.35)
    np.testing.assert_array_equal(flipped_image, np.flip(sample_image, axis=1))


def test_brightness_reduction_preserves_shape_and_dtype(
    sample_image: np.ndarray,
) -> None:
    reduced = brightness_reduction(sample_image.copy())

    assert reduced.shape == sample_image.shape
    assert reduced.dtype == np.uint8
    assert reduced.min() >= 0
    assert reduced.max() <= 255


def test_translation_preserves_dimensions(sample_image: np.ndarray) -> None:
    translated_image, translated_angle = translation(
        sample_image.copy(),
        0.1,
        x_translation_range=(-5, 5),
        y_translation_range=(-2, 2),
    )

    assert translated_image.shape == sample_image.shape
    assert translated_angle != 0.1


def test_top_bottom_crop_uses_expected_region(sample_image: np.ndarray) -> None:
    cropped = top_bottom_crop(sample_image)

    assert cropped.shape == (95, 320, 3)


def test_image_preprocessing_returns_model_ready_frame(
    sample_image: np.ndarray,
) -> None:
    processed = image_preprocessing(sample_image)

    assert processed.shape == (66, 200, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_prepare_driving_log_normalizes_camera_paths(tmp_path: Path) -> None:
    image_dir = tmp_path / "IMG"
    image_dir.mkdir()

    image_path = image_dir / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((160, 320, 3), dtype=np.uint8))

    csv_path = tmp_path / "driving_log.csv"
    csv_path.write_text(
        "center_camera,left_camera,right_camera,steering_angle,throttle,brake,speed\n"
        "IMG/frame.jpg,IMG/frame.jpg,IMG/frame.jpg,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    dataframe = prepare_driving_log(str(csv_path))

    assert len(dataframe) == 1
    assert dataframe.loc[0, "center_camera"] == str(image_path.resolve())


def test_batch_generator_yields_expected_shapes(tmp_path: Path) -> None:
    image_dir = tmp_path / "IMG"
    image_dir.mkdir()

    image_path = image_dir / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((160, 320, 3), dtype=np.uint8))

    dataframe = pd.DataFrame(
        {
            "center_camera": [str(image_path.resolve())],
            "left_camera": [str(image_path.resolve())],
            "right_camera": [str(image_path.resolve())],
            "steering_angle": [0.0],
            "throttle": [0.0],
            "brake": [0.0],
            "speed": [0.0],
        }
    )

    images, angles = next(batch_generator(dataframe, batch_size=2, training=False))

    assert images.shape == (2, 66, 200, 3)
    assert angles.shape == (2,)
    assert images.dtype == np.float32


def test_tensorflow_model_builds_and_predicts() -> None:
    _tensorflow()
    from self_driving_car.model import nvidia_model

    model = nvidia_model()
    output = model.predict(np.zeros((1, 66, 200, 3), dtype=np.float32), verbose=0)

    assert output.shape == (1, 1)
    assert model.count_params() == 252219
