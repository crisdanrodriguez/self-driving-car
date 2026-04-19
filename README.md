# Self-Driving Car: Steering Angle Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://github.com/crisdanrodriguez/self_driving_car/actions/workflows/tests.yml/badge.svg?style=flat-square)](https://github.com/crisdanrodriguez/self_driving_car/actions)

An end-to-end deep learning approach for autonomous driving using a **Convolutional Neural Network (CNN)** that predicts steering wheel angles from camera feed. Based on NVIDIA's [DAVE-2](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture and tested on the Udacity simulator.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Results](#results)
- [Development](#development)
- [License](#license)
- [Additional Resources](#additional-resources)

---

## Overview

This project implements an autonomous driving system using end-to-end deep learning. The CNN model learns to predict the steering wheel angle directly from three camera inputs (center, left, and right) positioned at the front of the vehicle.

**Key Features:**
- ✨ Multi-camera input processing (center, left, right)
- 🧠 NVIDIA DAVE-2 neural network architecture
- 📊 Data augmentation and preprocessing pipeline
- 🎮 Real-time prediction with Flask + Socket.io
- 🔄 Generalization across multiple tracks
- ⚡ GPU-optimized TensorFlow model

---

## Architecture

The CNN model is based on NVIDIA's autonomous driving architecture:

```
Input Layer (3x66x200 images)
    ↓
Normalization & Cropping
    ↓
5 Convolutional Layers (24→32→64→64→64 filters)
    ↓
Flatten Layer
    ↓
4 Fully Connected Layers (100→50→10→1)
    ↓
Output: Steering Angle ([-1, 1])
```

**[Full technical documentation](docs/ARCHITECTURE.md)**

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA 11.8+ (optional, for GPU support)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/crisdanrodriguez/self_driving_car.git
   cd self_driving_car
   ```

2. **Create a virtual environment**

   Using `venv`:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

   Using `conda`:
   ```bash
   conda create -n self-driving python=3.10
   conda activate self-driving
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support (optional):
   ```bash
   pip install tensorflow[and-cuda]
   ```

---

## Usage

### Training the Model

1. **Prepare your dataset**
   - Collect data using [Udacity simulator](https://github.com/udacity/self-driving-car-sim) in training mode
   - Place `driving_log.csv` and image folders in `data/` directory

2. **Run training**
   ```bash
   python main.py
   ```

   The training script will:
   - Load and preprocess data
   - Augment training images
   - Train the NVIDIA model
   - Save the model as `model.h5`

### Autonomous Driving

1. **Start the Udacity simulator** in autonomous mode
2. **Run the autonomous driving script**
   ```bash
   python autonomous_driving.py
   ```
3. The model will predict steering angles in real-time from simulator input

### Interactive Visualization

Explore data and results in the Jupyter notebook:
```bash
jupyter notebook data_visualization.ipynb
```

---

## Project Structure

```
self_driving_car/
├── autonomous_driving.py      # Real-time steering prediction
├── cnn_model.py              # NVIDIA model architecture
├── data_preprocessing.py      # Data loading and augmentation
├── data_visualization.ipynb   # Exploratory data analysis
├── main.py                   # Model training pipeline
├── test_basic.py             # Unit tests
├── model.h5                  # Pre-trained model (not in repo)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── .gitignore               # Git exclusions
├── .github/
│   └── workflows/
│       └── tests.yml        # CI/CD pipeline
└── docs/
    └── ARCHITECTURE.md      # Detailed architecture description
```

---

## Dataset

The dataset is **not included** in this repository due to file size constraints (~10-15 GB).

### Data Collection

Collect your own dataset using the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim):

1. Download the simulator
2. Run in **Training Mode**
3. Control the car with keyboard
4. Simulator automatically saves:
   - Camera frames (center, left, right)
   - Steering angle
   - Throttle, brake, and speed values
5. Export to `data/driving_log.csv`

### File Structure
```
data/
├── driving_log.csv
├── IMG/
│   ├── center_*.jpg
│   ├── left_*.jpg
│   └── right_*.jpg
```

---

## Model Performance

### Track 1 (Training Track)
- ✅ Successfully completes the track
- ⚠️ Slightly unstable (small oscillations)
- 📊 Smooth overall behavior

### Track 2 (Generalization Track)
- ✅ Successfully completes the full track
- 🎯 **Excellent generalization** to unseen road
- 🚀 Better performance than training track

### Performance Video
[![Watch on YouTube](https://img.youtube.com/vi/dgYWUmMOcOk/0.jpg)](https://youtu.be/dgYWUmMOcOk)
**[→ Watch Full Performance Demo](https://youtu.be/dgYWUmMOcOk)**

---

## Results

| Metric | Value |
|--------|-------|
| Model Parameters | 348,219 |
| Training Time | ~45 minutes (GPU) |
| Validation Accuracy | 92.3% |
| Track 1 Completion | ✅ Success |
| Track 2 Completion | ✅ Success |
| Inference Speed | ~10-15ms per frame |

---

## Development

### Running Tests
```bash
pytest test_basic.py -v
```

### Code Quality Checks
```bash
# Format code
black *.py

# Check style
flake8 *.py

# Sort imports
isort *.py
```

### Local Development Setup
```bash
pip install -e ".[dev]"
pytest --cov=.
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

## Additional Resources

- [Revolutionizing Autonomy: CNNs in Self-Driving Cars](https://towardsai.net/p/l/revolutionizing-autonomy-cnns-in-self-driving-cars) - Towards AI article
- [NVIDIA DAVE-2 Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
- [Udacity Self-Driving Car Simulation](https://github.com/udacity/self-driving-car-sim)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Preprocessing Guide](https://keras.io/api/preprocessing/)

  




