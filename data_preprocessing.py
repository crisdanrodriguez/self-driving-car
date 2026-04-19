"""
Data Preprocessing and Augmentation Pipeline.

This module provides functions for:
    - Image preprocessing (cropping, resizing, normalization)
    - Data augmentation (flip, brightness, translation)
    - Batch generation for model training and validation

The preprocessing pipeline ensures images are:
    - Cropped to region of interest (66x200)
    - Converted to YUV color space
    - Normalized to [0, 1] range
    - Ready for CNN input

Augmentation strategy:
    - Horizontal flipping with steering angle reversal
    - Random brightness modification
    - Random translation with steering angle adjustment
    - Multi-camera angle calibration (left: +0.25, center: 0, right: -0.25)
"""

import numpy as np
import cv2
from cv2 import cvtColor


def horizontal_flip(image, steering_angle):
    """
    Flip image horizontally and reverse steering angle.
    
    This augmentation technique helps the model generalize by
    providing symmetric training examples.
    
    Args:
        image (np.ndarray): Input image in RGB format
        steering_angle (float): Associated steering angle
        
    Returns:
        tuple: (flipped_image, reversed_steering_angle)
        - flipped_image (np.ndarray): Horizontally flipped image
        - steering_angle (float): Negated steering angle
        
    Example:
        >>> img, angle = horizontal_flip(image, 0.5)
        >>> assert angle == -0.5
    """
    flipped_image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    
    return flipped_image, steering_angle


def brightness_reduction(image):
    """
    Randomly modify image brightness to simulate different lighting conditions.
    
    Converts to HSV color space, modifies the V (value) channel,
    then converts back to RGB.
    
    Args:
        image (np.ndarray): Input image in RGB format (0-255)
        
    Returns:
        np.ndarray: Brightness-modified image in RGB format
        
    Technical Details:
        - Converts RGB → HSV (HSV is better for brightness manipulation)
        - Multiplies V channel by random factor in range [0.4, 0.8]
        - Clips values to [0, 255]
        - Converts back HSV → RGB
        
    Example:
        >>> modified = brightness_reduction(image)
        >>> assert modified.shape == image.shape
    """
    # Convert RGB to HSV for easier brightness manipulation
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hsv = np.array(image_hsv, dtype=np.float64)
    
    # Random brightness factor: range [0.4, 0.8]
    random_brightness = 0.8 - np.random.uniform(0, 0.4)
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * random_brightness
    
    # Ensure values don't exceed 255
    image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255
    
    # Convert back to RGB
    image_hsv = np.array(image_hsv, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    return image_rgb


def translation(image, steering_angle, x_translation_range=[-60, 60], y_translation_range=[-20, 20]):
    """
    Randomly translate image and adjust steering angle proportionally.
    
    Horizontal translation simulates the car being off-center and
    requires steering adjustment proportional to the shift.
    
    Args:
        image (np.ndarray): Input image
        steering_angle (float): Associated steering angle
        x_translation_range (list): Min and max horizontal pixel shift
        y_translation_range (list): Min and max vertical pixel shift
        
    Returns:
        tuple: (translated_image, adjusted_steering_angle)
        - translated_image (np.ndarray): Translated image (same size)
        - steering_angle (float): Adjusted steering angle
        
    Technical Details:
        - Horizontal shift coefficient: 0.0035 per pixel
        - Vertical shift is independent of steering
        - Uses affine transformation for smooth warping
        
    Example:
        >>> translated_img, adj_angle = translation(image, 0.0, x_translation_range=[-30, 30])
    """
    height, width = image.shape[0], image.shape[1]
    
    # Random translation values
    x_translation = np.random.randint(x_translation_range[0], x_translation_range[1])
    y_translation = np.random.randint(y_translation_range[0], y_translation_range[1])
    
    # Adjust steering angle: more left shift = more right steering needed
    steering_angle += x_translation * 0.0035
    
    # Create and apply translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return translated_image, steering_angle


def top_bottom_crop(image):
    """
    Crop top and bottom portions of image (hood and sky).
    
    Removes ~40 pixels from top (sky) and ~25 pixels from bottom (hood),
    keeping only the road region of interest.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Cropped image (95 pixels height instead of 160+)
        
    Example:
        >>> cropped = top_bottom_crop(image)
        >>> assert cropped.shape[0] == 95  # 160 - 40 - 25
    """
    # Crop: skip first 40 rows and last 25 rows
    cropped_image = image[40:135, :]
    return cropped_image


def augment_image(df):
    """
    Apply full augmentation pipeline to a random image from dataframe.
    
    Combines multiple augmentation techniques:
    1. Random camera selection (left, center, right)
    2. Steering angle calibration based on camera
    3. Random translation (50% probability)
    4. Random brightness (50% probability)
    5. Random horizontal flip (50% probability)
    
    Args:
        df (pd.DataFrame): Row with columns: center_camera, left_camera, 
                          right_camera, steering_angle
        
    Returns:
        tuple: (augmented_image, adjusted_steering_angle)
        
    Camera Calibration:
        - Center: 0.0 angle offset
        - Left: +0.25 angle offset (turn right to stay centered)
        - Right: -0.25 angle offset (turn left to stay centered)
        
    Example:
        >>> image, angle = augment_image(data_row)
    """
    # Randomly select camera (0: center, 1: left, 2: right)
    camera_side = np.random.randint(3)
    
    # Steering angle calibration for each camera position
    if camera_side == 0:
        image_path = df.iloc[0]['center_camera'].strip()
        angle_calibration = 0.0
    elif camera_side == 1:
        image_path = df.iloc[0]['left_camera'].strip()
        angle_calibration = 0.25
    else:  # camera_side == 2
        image_path = df.iloc[0]['right_camera'].strip()
        angle_calibration = -0.25
    
    steering_angle = df.iloc[0]['steering_angle'] + angle_calibration
    
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply random augmentations
    if np.random.rand() < 0.5:
        image, steering_angle = translation(image, steering_angle)
    
    if np.random.rand() < 0.5:
        image = brightness_reduction(image)
    
    if np.random.rand() < 0.5:
        image, steering_angle = horizontal_flip(image, steering_angle)
    
    return image, steering_angle


def image_preprocessing(image):
    """
    Preprocess image for model input.
    
    Pipeline:
    1. Crop top/bottom (region of interest)
    2. Convert RGB → YUV (better for neural networks)
    3. Resize to (200, 66)
    4. Normalize to [0, 1] range
    
    Args:
        image (np.ndarray): Input image in RGB format (0-255)
        
    Returns:
        np.ndarray: Preprocessed image ready for model input
        - Shape: (66, 200, 3)
        - Values: [0, 1] (normalized)
        - Color space: YUV
        
    Example:
        >>> processed = image_preprocessing(raw_image)
        >>> assert processed.shape == (66, 200, 3)
        >>> assert processed.min() >= 0 and processed.max() <= 1
    """
    # Crop to region of interest
    image = top_bottom_crop(image)
    
    # Convert RGB to YUV (Y channel has more information about edges)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Resize to model input dimensions
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    return image


def batch_generator(df, batch_size, training_flag):
    """
    Generate batches of preprocessed images and steering angles.
    
    Infinite generator that yields batches of data for training/validation.
    Training batches include augmentation, validation batches do not.
    
    Args:
        df (pd.DataFrame): Dataframe with columns: center_camera, left_camera,
                          right_camera, steering_angle, throttle, brake, speed
        batch_size (int): Number of samples per batch
        training_flag (bool): If True, apply augmentation; if False, no augmentation
        
    Yields:
        tuple: (images_batch, angles_batch)
        - images_batch (np.ndarray): Shape (batch_size, 66, 200, 3)
        - angles_batch (np.ndarray): Shape (batch_size,)
        
    Example:
        >>> gen = batch_generator(train_df, batch_size=64, training_flag=True)
        >>> images, angles = next(gen)
        >>> assert images.shape == (64, 66, 200, 3)
    """
    while True:
        images_batch = []
        steering_angles_batch = []
        
        for _ in range(batch_size):
            # Select random sample from dataframe
            index = np.random.randint(0, len(df) - 1)
            
            if training_flag:
                # Training: apply full augmentation
                image, steering_angle = augment_image(df.iloc[[index]])
            else:
                # Validation: random camera but no augmentation
                camera_side = np.random.randint(3)
                
                if camera_side == 0:
                    image_path = df.iloc[index]['center_camera'].strip()
                    angle_calibration = 0.0
                elif camera_side == 1:
                    image_path = df.iloc[index]['left_camera'].strip()
                    angle_calibration = 0.25
                else:
                    image_path = df.iloc[index]['right_camera'].strip()
                    angle_calibration = -0.25
                
                image = cv2.imread(image_path)
                steering_angle = df.iloc[index]['steering_angle'] + angle_calibration
            
            # Preprocess image
            image = image_preprocessing(image)
            
            # Add to batch
            images_batch.append(image)
            steering_angles_batch.append(steering_angle)
        
        # Yield batch
        yield (np.asarray(images_batch), np.asarray(steering_angles_batch))
