# Self-Driving Car Steering Angle Prediction

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?style=flat-square)](https://www.python.org/downloads/)
[![Tests](https://github.com/crisdanrodriguez/self-driving_car/actions/workflows/tests.yml/badge.svg?style=flat-square)](https://github.com/crisdanrodriguez/self-driving_car/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange?style=flat-square)](https://www.tensorflow.org/)

End-to-end steering-angle prediction for the Udacity self-driving car simulator using a CNN inspired by NVIDIA's DAVE-2 architecture. The repository includes data preprocessing, model training, realtime inference, automated tests, and GitHub collaboration templates.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Overview

This project trains a convolutional neural network to predict steering commands directly from front-facing camera frames. It is designed around the Udacity simulator workflow:

- preprocess and augment simulator images
- train a regression model on `driving_log.csv`
- load the trained model for realtime steering inference
- validate repository health through automated checks

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

Record driving sessions with the Udacity simulator and place the exported files under `data/`:

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

### 4. Explore the Notebook

```bash
jupyter notebook data_visualization.ipynb
```

## Project Structure

```text
self-driving_car/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/tests.yml
│   └── pull_request_template.md
├── docs/
│   └── ARCHITECTURE.md
├── autonomous_driving.py
├── cnn_model.py
├── data_preprocessing.py
├── data_visualization.ipynb
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── test_basic.py
├── CONTRIBUTING.md
└── README.md
```

## Results

The repository includes the full training and inference workflow, but model quality depends on the dataset you collect. Because steering-angle prediction is a regression task, meaningful evaluation should focus on:

- validation loss and MAE during training
- qualitative behavior in the simulator
- robustness on unseen tracks and lighting conditions

A reference demo asset is included in the repository:

- [Project presentation PDF](Self-Driving%20Car,%20Predicting%20Steering%20Wheel%20Angle.pdf)
- [Architecture notes](docs/ARCHITECTURE.md)

## Documentation

- [Architecture documentation](docs/ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)
- [Contribution guide](CONTRIBUTING.md)

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
