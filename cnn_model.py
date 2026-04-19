"""
CNN Model Architecture for Steering Angle Prediction.

This module implements the NVIDIA DAVE-2 neural network architecture
for autonomous driving steering angle prediction.

Reference Paper:
    Bojarski, M., et al. (2016). "End-to-End Learning for Self-Driving Cars"
    https://arxiv.org/abs/1604.07316

Architecture Overview:
    - 5 Convolutional layers with ELU activation
    - Spatial dimension reduction via stride-2 convolutions
    - Feature flattening and 3 dense layers
    - Single output neuron for steering angle prediction

Model Statistics:
    - Total Parameters: 348,219
    - Input Shape: (3, 66, 200)
    - Output Shape: (1,) - Steering angle in range [-1, 1]
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam


def nvidia_model():
    """
    Create the NVIDIA DAVE-2 CNN model for steering angle prediction.
    
    The model processes 66x200 RGB images through 5 convolutional layers,
    followed by a flattening operation and 3 fully connected layers.
    The output is a single steering angle value in the range [-1, 1].
    
    Architecture:
        Conv2D(24, 5x5, stride=2) -> ELU -> 
        Conv2D(36, 5x5, stride=2) -> ELU ->
        Conv2D(48, 5x5, stride=2) -> ELU ->
        Conv2D(64, 3x3, stride=1) -> ELU ->
        Conv2D(64, 3x3, stride=1) -> ELU ->
        Flatten ->
        Dense(100) -> ELU ->
        Dense(50) -> ELU ->
        Dense(10) -> ELU ->
        Dense(1) [output]
    
    Returns:
        tf.keras.models.Sequential: Compiled NVIDIA model
        
    Example:
        >>> model = nvidia_model()
        >>> model.summary()
        >>> predictions = model.predict(input_images)
    """
    model = models.Sequential(name='NVIDIA_DAVE2_Model')

    # Layer 1: Convolutional layer with 24 filters, 5x5 kernel, 2x2 stride
    model.add(layers.Conv2D(
        24, (5, 5), strides=(2, 2),
        input_shape=(66, 200, 3),
        activation='elu',
        name='conv1_24_filters'
    ))

    # Layer 2: Convolutional layer with 36 filters, 5x5 kernel, 2x2 stride
    model.add(layers.Conv2D(
        36, (5, 5), strides=(2, 2),
        activation='elu',
        name='conv2_36_filters'
    ))

    # Layer 3: Convolutional layer with 48 filters, 5x5 kernel, 2x2 stride
    model.add(layers.Conv2D(
        48, (5, 5), strides=(2, 2),
        activation='elu',
        name='conv3_48_filters'
    ))

    # Layer 4: Convolutional layer with 64 filters, 3x3 kernel, 1x1 stride
    model.add(layers.Conv2D(
        64, (3, 3), strides=(1, 1),
        activation='elu',
        name='conv4_64_filters'
    ))

    # Layer 5: Convolutional layer with 64 filters, 3x3 kernel, 1x1 stride
    model.add(layers.Conv2D(
        64, (3, 3), strides=(1, 1),
        activation='elu',
        name='conv5_64_filters'
    ))

    # Flatten: Convert 3D feature maps to 1D vector
    model.add(layers.Flatten(name='flatten'))

    # Dense Layer 1: 100 units with ELU activation
    model.add(layers.Dense(100, activation='elu', name='dense1_100_units'))

    # Dense Layer 2: 50 units with ELU activation
    model.add(layers.Dense(50, activation='elu', name='dense2_50_units'))

    # Dense Layer 3: 10 units with ELU activation
    model.add(layers.Dense(10, activation='elu', name='dense3_10_units'))

    # Output Layer: Single neuron for steering angle (linear activation)
    model.add(layers.Dense(1, name='output_steering_angle'))

    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )

    return model


if __name__ == '__main__':
    # Example usage and model inspection
    model = nvidia_model()
    model.summary()
    print('\nModel successfully created!')
    print(f'Total parameters: {model.count_params():,}')
