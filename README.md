# Self-Driving Car Steering Angle Prediction

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?style=flat-square)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange?style=flat-square)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://github.com/crisdanrodriguez/self-driving_car/actions/workflows/tests.yml/badge.svg?style=flat-square)](https://github.com/crisdanrodriguez/self-driving_car/actions/workflows/tests.yml)

Compact end-to-end steering-angle prediction project for the Udacity self-driving car simulator, built with TensorFlow/Keras and organized as a clean, installable Python package.

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

This repository trains a convolutional neural network to predict steering angles directly from front-facing simulator frames. It focuses on a minimal but professional layout: core training and inference code, lightweight tests, reproducible setup files, and curated supporting documentation.

Project profile:

- Type: Machine learning / computer vision
- Language: Python
- Framework: TensorFlow + Keras
- Interface: CLI training flow and Socket.IO inference server for the Udacity simulator

What the repository currently includes:

- Udacity-style `driving_log.csv` preprocessing and camera-path normalization
- On-the-fly image augmentation and batch generation
- NVIDIA DAVE-2 inspired regression model
- Realtime simulator bridge for autonomous steering inference
- Basic automated validation through tests and CI

## Installation

Prerequisites:

- Python `3.10` or `3.11`
- `pip`
- Optional TensorFlow-compatible acceleration if you plan to train locally

Setup:

```bash
git clone https://github.com/crisdanrodriguez/self-driving_car.git
cd self-driving_car
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

## Usage

### 1. Prepare the dataset

Export a driving session from the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) and place the files under `data/`:

```text
data/
├── driving_log.csv
└── IMG/
    ├── center_*.jpg
    ├── left_*.jpg
    └── right_*.jpg
```

The preprocessing pipeline resolves Udacity-style relative paths and normalizes Windows-style path separators when needed.

### 2. Train the model

```bash
python -m self_driving_car.train \
  --data-path data/driving_log.csv \
  --model-output artifacts/model.h5
```

Useful commands:

```bash
python -m self_driving_car.train --help
python -m self_driving_car.train --epochs 10 --batch-size 64 --steps-per-epoch 300
```

### 3. Run autonomous inference

Start the Udacity simulator in autonomous mode, then run:

```bash
python -m self_driving_car.inference --model-path artifacts/model.h5 --port 4567
```

Console scripts are also available after installation:

```bash
train-steering-model --help
run-simulator-bridge --help
```

## Project Structure

```text
self-driving_car/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── pull_request_template.md
│   └── workflows/tests.yml
├── docs/
│   ├── documentation/
│   │   ├── architecture.md
│   │   └── steering-angle-presentation.pdf
│   └── results/
│       └── steering-angle-presentation.pptx
├── notebooks/
│   └── data-visualization.ipynb
├── src/
│   └── self_driving_car/
│       ├── __init__.py
│       ├── inference.py
│       ├── model.py
│       ├── preprocessing.py
│       └── train.py
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Results

- [Performance demo on YouTube](https://www.youtube.com/watch?v=dgYWUmMOcOk)
- [Project presentation source (PPTX)](docs/results/steering-angle-presentation.pptx)

## Documentation

- [Architecture notes](docs/documentation/architecture.md)
- [Project presentation (PDF)](docs/documentation/steering-angle-presentation.pdf)
- [Data visualization notebook](notebooks/data-visualization.ipynb)
- [Toward AI article: Revolutionizing Autonomy: CNNs in Self-Driving Cars](https://towardsai.net/p/l/revolutionizing-autonomy-cnns-in-self-driving-cars)

## Development

Run the local checks with:

```bash
pytest -q
black --check .
isort --check-only .
flake8 .
```

The GitHub Actions workflow runs the same validation on pushes and pull requests to `main`.

## License

This project is licensed under the [MIT License](LICENSE).

## AI Assistance and Last Updated

This repository was refined with AI-assisted support using OpenAI Codex, with changes reviewed and applied in the local workspace.

Last updated: April 19, 2026
