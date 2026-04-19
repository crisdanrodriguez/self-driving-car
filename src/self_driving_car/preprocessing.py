"""Data preprocessing and augmentation utilities for the driving dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAMERA_COLUMNS = ("center_camera", "left_camera", "right_camera")
DRIVING_LOG_COLUMNS = [
    "center_camera",
    "left_camera",
    "right_camera",
    "steering_angle",
    "throttle",
    "brake",
    "speed",
]


def horizontal_flip(
    image: np.ndarray, steering_angle: float
) -> tuple[np.ndarray, float]:
    """Flip an image horizontally and reverse the steering angle."""

    return cv2.flip(image, 1), -steering_angle


def brightness_reduction(image: np.ndarray) -> np.ndarray:
    """Randomly reduce image brightness to improve lighting robustness."""

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    random_brightness = 0.8 - np.random.uniform(0.0, 0.4)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] * random_brightness, 0, 255)
    return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def translation(
    image: np.ndarray,
    steering_angle: float,
    x_translation_range: tuple[int, int] = (-60, 60),
    y_translation_range: tuple[int, int] = (-20, 20),
) -> tuple[np.ndarray, float]:
    """Translate an image and adjust the steering angle proportionally."""

    height, width = image.shape[:2]
    x_translation = np.random.randint(
        x_translation_range[0], x_translation_range[1] + 1
    )
    y_translation = np.random.randint(
        y_translation_range[0], y_translation_range[1] + 1
    )

    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    adjusted_angle = steering_angle + (x_translation * 0.0035)
    return translated_image, adjusted_angle


def top_bottom_crop(image: np.ndarray) -> np.ndarray:
    """Crop the sky and car hood, keeping the road-focused region of interest."""

    return image[40:135, :]


def resolve_image_path(image_path: str, data_root: Path) -> Path:
    """Resolve image paths from Udacity driving logs across operating systems."""

    raw_path = image_path.strip().replace("\\", "/")
    candidates = [
        Path(raw_path),
        data_root / raw_path,
        data_root / "IMG" / Path(raw_path).name,
    ]

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Unable to resolve image path: {image_path}")


def prepare_driving_log(data_path: str) -> pd.DataFrame:
    """Load and normalize a Udacity-style driving log for training."""

    csv_path = Path(data_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Driving log not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    normalized_columns = [str(column).strip().lower() for column in dataframe.columns]
    if normalized_columns == [column.lower() for column in DRIVING_LOG_COLUMNS]:
        dataframe.columns = DRIVING_LOG_COLUMNS
    elif len(dataframe.columns) == len(DRIVING_LOG_COLUMNS):
        dataframe.columns = DRIVING_LOG_COLUMNS
    else:
        dataframe = pd.read_csv(csv_path, names=DRIVING_LOG_COLUMNS)

    data_root = csv_path.parent
    for column in CAMERA_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .map(lambda path: str(resolve_image_path(path, data_root)))
        )

    dataframe["steering_angle"] = pd.to_numeric(
        dataframe["steering_angle"], errors="coerce"
    )
    dataframe = dataframe.dropna(subset=["steering_angle"]).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("The driving log does not contain valid steering samples.")

    return dataframe


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and return it in RGB format."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def augment_image(row: pd.Series) -> tuple[np.ndarray, float]:
    """Apply camera selection and random augmentation to a sample."""

    camera_side = np.random.randint(3)
    camera_column = CAMERA_COLUMNS[camera_side]
    angle_offsets = {"center_camera": 0.0, "left_camera": 0.25, "right_camera": -0.25}

    image = load_rgb_image(row[camera_column])
    steering_angle = float(row["steering_angle"]) + angle_offsets[camera_column]

    if np.random.rand() < 0.5:
        image, steering_angle = translation(image, steering_angle)
    if np.random.rand() < 0.5:
        image = brightness_reduction(image)
    if np.random.rand() < 0.5:
        image, steering_angle = horizontal_flip(image, steering_angle)

    return image, steering_angle


def image_preprocessing(image: np.ndarray) -> np.ndarray:
    """Crop, convert color space, resize, and normalize a camera frame."""

    image = top_bottom_crop(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def batch_generator(
    dataframe: pd.DataFrame,
    batch_size: int,
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Yield infinite batches of preprocessed images and steering angles."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if dataframe.empty:
        raise ValueError("The provided dataframe is empty.")

    angle_offsets = {"center_camera": 0.0, "left_camera": 0.25, "right_camera": -0.25}

    while True:
        images_batch: list[np.ndarray] = []
        steering_angles_batch: list[float] = []

        for _ in range(batch_size):
            index = np.random.randint(0, len(dataframe))
            row = dataframe.iloc[index]

            if training:
                image, steering_angle = augment_image(row)
            else:
                camera_column = CAMERA_COLUMNS[np.random.randint(3)]
                image = load_rgb_image(row[camera_column])
                steering_angle = (
                    float(row["steering_angle"]) + angle_offsets[camera_column]
                )

            images_batch.append(image_preprocessing(image))
            steering_angles_batch.append(steering_angle)

        yield np.asarray(images_batch), np.asarray(
            steering_angles_batch, dtype=np.float32
        )
