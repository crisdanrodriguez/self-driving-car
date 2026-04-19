# Self-Driving Car Steering Angle Prediction

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?style=flat-square)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange?style=flat-square)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://github.com/crisdanrodriguez/self-driving_car/actions/workflows/tests.yml/badge.svg?style=flat-square)](https://github.com/crisdanrodriguez/self-driving_car/actions/workflows/tests.yml)

End-to-end steering-angle prediction for the Udacity self-driving car simulator using a CNN inspired by NVIDIA's DAVE-2 architecture. The repository focuses on the essential application code, reproducible setup files, automated tests, and core documentation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)
- [AI Assistance and Last Updated](#ai-assistance-and-last-updated)

## Overview

This project trains a convolutional neural network to predict steering commands directly from front-facing camera frames. It is designed around the Udacity simulator workflow:

- Preprocess and augment simulator images
- Train a regression model on `driving_log.csv`
- Load the trained model for realtime steering inference
- Validate repository health through automated checks

### Project Type

- Language: Python
- Domain: Machine Learning / Computer Vision
- Runtime: TensorFlow + Keras
- Interface: CLI scripts and Socket.IO server for the Udacity simulator

### Supported Environment

- Python `3.10` or `3.11`
- Linux: supported for training and CI
- macOS Apple Silicon: supported through `tensorflow-macos`
- Windows: preprocessing utilities and project structure are supported; full training/inference compatibility depends on TensorFlow availability

## Installation

### Prerequisites

- Python `3.10` or `3.11`
- `pip`
- Optional GPU acceleration configured for TensorFlow

### Setup

```bash
git clone https://github.com/crisdanrodriguez/self-driving_car.git
cd self-driving_car
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Usage

### 1. Prepare the Dataset

Record driving sessions with the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) and place the exported files under `data/`:

```text
data/
├── driving_log.csv
└── IMG/
    ├── center_*.jpg
    ├── left_*.jpg
    └── right_*.jpg
```

The training pipeline normalizes Udacity-style camera paths automatically, including Windows-style paths stored inside `driving_log.csv`.

### 2. Train the Model

```bash
python main.py --data-path data/driving_log.csv --model-output model.h5
```

Useful options:

```bash
python main.py --help
python main.py --epochs 10 --batch-size 64 --steps-per-epoch 300
```

### 3. Run Autonomous Driving

Start the Udacity simulator in autonomous mode and then launch the inference bridge:

```bash
python autonomous_driving.py --model-path model.h5 --port 4567
```

## Project Structure

```text
self-driving-car/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/tests.yml
│   └── pull_request_template.md
├── docs/
│   └── ARCHITECTURE.md
├── autonomous_driving.py
├── cnn_model.py
├── data_preprocessing.py
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── test_basic.py
└── README.md
```

## Results

## Results

Project results and supporting material are available here:

- [Performance demo on YouTube](https://www.youtube.com/watch?v=dgYWUmMOcOk)
- [Project presentation (PDF)](docs/Self-Driving%20Car,%20Predicting%20Steering%20Wheel%20Angle.pdf)
- [Project presentation source (PPTX)](docs/Self-Driving%20Car,%20Predicting%20Steering%20Wheel%20Angle.pptx)
- [Toward AI article: Revolutionizing Autonomy: CNNs in Self-Driving Cars](https://towardsai.net/p/l/revolutionizing-autonomy-cnns-in-self-driving-cars)


## Documentation

- [Architecture documentation](docs/ARCHITECTURE.md)
- [Project presentation (PDF)](docs/Self-Driving%20Car,%20Predicting%20Steering%20Wheel%20Angle.pdf)
- [Project presentation source (PPTX)](docs/Self-Driving%20Car,%20Predicting%20Steering%20Wheel%20Angle.pptx)
- [Toward AI article: Revolutionizing Autonomy: CNNs in Self-Driving Cars](https://towardsai.net/p/l/revolutionizing-autonomy-cnns-in-self-driving-cars)


## Development

Run the local quality checks with:

```bash
pytest -q
black --check .
isort --check-only .
flake8 .
```

GitHub Actions runs the test workflow automatically on pushes and pull requests to `main`.

## License

This project is licensed under the [MIT License](LICENSE).

## AI Assistance and Last Updated

This repository's GitHub professionalization was completed with AI-assisted support
using OpenAI Codex, with the final changes reviewed and applied in the repository
workspace.

Last updated: April 19, 2026
