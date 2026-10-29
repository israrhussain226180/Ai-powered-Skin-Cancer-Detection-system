# Skin Cancer Classification Project

A comprehensive deep learning pipeline for skin cancer detection and classification using the ISIC dataset.

## Project Structure

- `data/`: Contains raw, processed, and augmented datasets.
- `notebooks/`: Jupyter notebooks for EDA, experiments, and evaluation.
- `src/`: Core Python source code for data loading, model architecture, training, and utilities.
- `models/`: Directory for saved model weights and checkpoints.
- `reports/`: Generated figures and performance results.
- `webapp/`: Django-based web application for deployment.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your data in `data/raw/`.
3. Run the training pipeline:
   ```bash
   python main.py
   ```
