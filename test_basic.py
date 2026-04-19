"""
Basic unit tests for the self-driving car CNN model.

Tests cover:
- Model architecture validation
- Data preprocessing functions
- Image augmentation pipeline
- Model inference capability
"""

import unittest
import numpy as np
import tempfile
import os

# Import modules to test
from cnn_model import nvidia_model
from data_preprocessing import (
    horizontal_flip,
    brightness_reduction,
    translation,
    top_bottom_crop,
    augment_image
)


class TestNVIDIAModel(unittest.TestCase):
    """Test cases for the NVIDIA CNN model architecture."""

    def setUp(self):
        """Initialize test fixtures."""
        self.model = nvidia_model()

    def test_model_creation(self):
        """Test that model is created successfully."""
        self.assertIsNotNone(self.model)

    def test_model_summary(self):
        """Test that model has expected structure."""
        # Model should have multiple layers
        self.assertGreater(len(self.model.layers), 5)

    def test_model_output_shape(self):
        """Test that model produces correct output shape."""
        # Input shape: (batch_size, 66, 200, 3)
        test_input = np.random.randn(1, 66, 200, 3).astype(np.float32)
        output = self.model.predict(test_input, verbose=0)
        # Output should be (1, 1) for steering angle
        self.assertEqual(output.shape, (1, 1))

    def test_model_output_range(self):
        """Test that model output is in reasonable steering angle range."""
        test_input = np.random.randn(5, 66, 200, 3).astype(np.float32)
        output = self.model.predict(test_input, verbose=0)
        # Steering angles should typically be in [-1, 1]
        self.assertTrue(np.all(output >= -2.0) and np.all(output <= 2.0))

    def test_model_batch_processing(self):
        """Test that model can handle multiple samples."""
        batch_sizes = [1, 5, 10, 32]
        for batch_size in batch_sizes:
            test_input = np.random.randn(batch_size, 66, 200, 3).astype(np.float32)
            output = self.model.predict(test_input, verbose=0)
            self.assertEqual(output.shape[0], batch_size)


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functions."""

    def setUp(self):
        """Create test images."""
        # Create a test image (66, 200, 3) in RGB format
        self.test_image = np.random.randint(0, 256, (66, 200, 3), dtype=np.uint8)
        self.test_steering = 0.5

    def test_horizontal_flip(self):
        """Test horizontal flip augmentation."""
        flipped_image, flipped_angle = horizontal_flip(
            self.test_image.copy(),
            self.test_steering
        )
        
        # Check shapes are preserved
        self.assertEqual(flipped_image.shape, self.test_image.shape)
        # Check steering angle is negated
        self.assertAlmostEqual(flipped_angle, -self.test_steering)
        # Check image is actually flipped
        np.testing.assert_array_almost_equal(
            flipped_image,
            np.flip(self.test_image, axis=1)
        )

    def test_brightness_reduction(self):
        """Test brightness modification."""
        modified_image = brightness_reduction(self.test_image.copy())
        
        # Check shape is preserved
        self.assertEqual(modified_image.shape, self.test_image.shape)
        # Check data type is uint8
        self.assertEqual(modified_image.dtype, np.uint8)
        # Check image values are valid
        self.assertTrue(np.all(modified_image >= 0) and np.all(modified_image <= 255))

    def test_translation(self):
        """Test image translation."""
        translated_image, translated_angle = translation(
            self.test_image.copy(),
            self.test_steering
        )
        
        # Check shape is preserved
        self.assertEqual(translated_image.shape, self.test_image.shape)
        # Check steering angle is adjusted
        self.assertNotEqual(translated_angle, self.test_steering)

    def test_top_bottom_crop(self):
        """Test image cropping."""
        cropped_image = top_bottom_crop(self.test_image.copy())
        
        # Original image: (66, 200, 3)
        # Expected: (66-40-25, 200, 3) = (1, 200, 3)
        expected_height = 66 - 40 - 25
        self.assertEqual(cropped_image.shape, (expected_height, 200, 3))


class TestModelInference(unittest.TestCase):
    """Test cases for model inference."""

    def setUp(self):
        """Initialize model for inference."""
        self.model = nvidia_model()

    def test_inference_with_zero_input(self):
        """Test model inference with zero input."""
        test_input = np.zeros((1, 66, 200, 3), dtype=np.float32)
        output = self.model.predict(test_input, verbose=0)
        self.assertIsNotNone(output)

    def test_inference_with_normalized_input(self):
        """Test model inference with normalized input."""
        test_input = np.random.randn(1, 66, 200, 3).astype(np.float32)
        test_input = test_input / 255.0  # Normalize to [0, 1]
        output = self.model.predict(test_input, verbose=0)
        self.assertIsNotNone(output)

    def test_deterministic_output(self):
        """Test that same input produces consistent output."""
        test_input = np.ones((1, 66, 200, 3), dtype=np.float32)
        outputs = [self.model.predict(test_input, verbose=0) for _ in range(3)]
        
        # All outputs should be identical
        for output in outputs[1:]:
            np.testing.assert_array_almost_equal(outputs[0], output)


class TestImportCompatibility(unittest.TestCase):
    """Test that all imports work correctly."""

    def test_imports(self):
        """Test that all required packages can be imported."""
        try:
            import tensorflow
            import numpy
            import pandas
            import cv2
            import sklearn
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import required package: {e}")

    def test_tensorflow_version(self):
        """Test TensorFlow version compatibility."""
        import tensorflow as tf
        version = tuple(map(int, tf.__version__.split('.')[:2]))
        # TensorFlow 2.0+
        self.assertGreaterEqual(version[0], 2)

    def test_numpy_array_operations(self):
        """Test basic NumPy operations."""
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        self.assertEqual(result, 3.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Initialize test fixtures."""
        self.model = nvidia_model()

    def test_empty_batch(self):
        """Test handling of empty batch."""
        test_input = np.empty((0, 66, 200, 3), dtype=np.float32)
        try:
            output = self.model.predict(test_input, verbose=0)
            self.assertEqual(output.shape[0], 0)
        except Exception:
            # Empty batch might raise an exception, which is acceptable
            pass

    def test_large_steering_angle(self):
        """Test handling of extreme steering angles."""
        extreme_angles = [-2.0, -1.5, 1.5, 2.0]
        for angle in extreme_angles:
            # Should not raise exception
            flipped, result = horizontal_flip(
                np.random.randint(0, 256, (66, 200, 3), dtype=np.uint8),
                angle
            )
            self.assertIsNotNone(flipped)


if __name__ == '__main__':
    unittest.main()
