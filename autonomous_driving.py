"""Realtime inference server for the Udacity self-driving simulator."""

from __future__ import annotations

import argparse
import base64
import os
from io import BytesIO
from pathlib import Path

import eventlet
import numpy as np
import socketio
from flask import Flask
from PIL import Image

from data_preprocessing import image_preprocessing

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sio = socketio.Server(async_mode="eventlet")
flask_app = Flask(__name__)
model = None
max_speed = 30.0


@sio.on("telemetry")
def telemetry(_sid: str, data: dict) -> None:
    """Receive simulator frames, predict steering, and send control commands."""

    if model is None:
        raise RuntimeError("The model must be loaded before starting telemetry.")

    try:
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        frame = np.asarray(image)
        processed_frame = np.array([image_preprocessing(frame)])

        steering = float(model.predict(processed_frame, verbose=0).squeeze())
        throttle = max(0.0, 1.0 - speed / max_speed)

        print(
            f"Steering: {steering:.4f} | Throttle: {throttle:.4f} | Speed: {speed:.2f}"
        )
        send_control(steering, throttle)
    except Exception as exc:
        print(f"Telemetry error: {exc}")
        send_control(0.0, 0.0)


@sio.on("connect")
def connect(_sid: str, _environ: dict) -> None:
    """Handle simulator connection events."""

    print("Simulator connected.")
    send_control(0.0, 0.0)


@sio.on("disconnect")
def disconnect(_sid: str) -> None:
    """Handle simulator disconnection events."""

    print("Simulator disconnected.")


def send_control(steering: float, throttle: float) -> None:
    """Send steering and throttle values back to the simulator."""

    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle),
        },
    )


def load_driving_model(model_path: str):
    """Load the trained Keras model lazily to keep imports lightweight."""

    from tensorflow.keras.models import load_model

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")
    return load_model(resolved_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the realtime server."""

    parser = argparse.ArgumentParser(
        description="Run realtime steering inference for the Udacity simulator.",
    )
    parser.add_argument(
        "--model-path",
        default="model.h5",
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for the Socket.IO server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4567,
        help="Port for the Socket.IO server.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=30.0,
        help="Reference speed used to compute throttle.",
    )
    return parser


def main() -> None:
    """Start the realtime inference server."""

    global model
    global max_speed

    args = build_parser().parse_args()
    if args.max_speed <= 0:
        raise ValueError("--max-speed must be greater than 0.")

    model = load_driving_model(args.model_path)
    max_speed = args.max_speed

    app = socketio.WSGIApp(sio, flask_app)
    print(f"Loaded model from {Path(args.model_path).expanduser().resolve()}")
    print(f"Starting simulator bridge on http://{args.host}:{args.port}")
    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app)


if __name__ == "__main__":
    main()
