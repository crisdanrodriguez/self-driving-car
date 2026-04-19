"""Realtime inference server for the Udacity self-driving simulator."""

from __future__ import annotations

import argparse
import base64
import os
from io import BytesIO
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_driving_model(model_path: str):
    """Load the trained Keras model lazily to keep imports lightweight."""

    from tensorflow.keras.models import load_model

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")
    return load_model(resolved_path)


def create_simulator_wsgi_app(driving_model, max_speed: float):
    """Create the Socket.IO WSGI application for realtime simulator inference."""

    import numpy as np
    import socketio
    from flask import Flask
    from PIL import Image

    from self_driving_car.preprocessing import image_preprocessing

    sio = socketio.Server(async_mode="eventlet")
    flask_app = Flask(__name__)

    def send_control(steering: float, throttle: float) -> None:
        """Send steering and throttle values back to the simulator."""

        sio.emit(
            "steer",
            data={
                "steering_angle": str(steering),
                "throttle": str(throttle),
            },
        )

    @sio.on("telemetry")
    def telemetry(_sid: str, data: dict) -> None:
        """Receive simulator frames, predict steering, and send control commands."""

        try:
            speed = float(data["speed"])
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            frame = np.asarray(image)
            processed_frame = np.array([image_preprocessing(frame)])

            steering = float(
                driving_model.predict(processed_frame, verbose=0).squeeze()
            )
            throttle = max(0.0, 1.0 - speed / max_speed)

            print(
                "Steering: "
                f"{steering:.4f} | Throttle: {throttle:.4f} | Speed: {speed:.2f}"
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

    return socketio.WSGIApp(sio, flask_app)


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

    args = build_parser().parse_args()
    if args.max_speed <= 0:
        raise ValueError("--max-speed must be greater than 0.")

    import eventlet

    model = load_driving_model(args.model_path)
    app = create_simulator_wsgi_app(model, args.max_speed)

    print(f"Loaded model from {Path(args.model_path).expanduser().resolve()}")
    print(f"Starting simulator bridge on http://{args.host}:{args.port}")
    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app)


if __name__ == "__main__":
    main()
