"""
Real-time autonomous driving using the trained CNN model with Udacity simulator.

This module communicates with the Udacity self-driving car simulator via WebSocket,
receives camera frames, predicts steering angles, and sends control commands back.

Original concept from: https://youtu.be/mVUrErF5xq8
Author: Murtaza's Workshop - Robotics and AI
Modified: Professional version with improvements
"""

import os
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model

from data_preprocessing import image_preprocessing

# Disable TensorFlow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global variables
sio = socketio.Server(async_mode='eventlet')
app = Flask(__name__)
model = None
MAX_SPEED = 30


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Handle telemetry data from the simulator.
    
    Args:
        sid: Session ID
        data: Dictionary containing speed and base64-encoded image
    """
    try:
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        
        # Preprocess image
        image = image_preprocessing(image)
        image = np.array([image])
        
        # Predict steering angle
        steering = float(model.predict(image, verbose=0))
        
        # Calculate throttle based on speed
        throttle = 1.0 - speed / MAX_SPEED
        
        print(f'Steering: {steering:.4f} | Throttle: {throttle:.4f} | Speed: {speed:.2f}')
        send_control(steering, throttle)
    except Exception as e:
        print(f'Error in telemetry: {e}')
        send_control(0, 0)


@sio.on('connect')
def connect(sid, environ):
    """Handle client connection."""
    print('✓ Simulator connected')
    send_control(0, 0)


@sio.on('disconnect')
def disconnect(sid):
    """Handle client disconnection."""
    print('✗ Simulator disconnected')


def send_control(steering, throttle):
    """
    Send control commands to the simulator.
    
    Args:
        steering (float): Steering angle [-1, 1]
        throttle (float): Throttle value [0, 1]
    """
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })


if __name__ == '__main__':
    print('Loading model...')
    model = load_model('model.h5')
    
    print('Starting server on http://localhost:4567')
    print('Connect the Udacity simulator to begin autonomous driving...\n')
    
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)