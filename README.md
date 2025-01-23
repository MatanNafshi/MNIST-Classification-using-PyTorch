# MNIST Digit Classification

This project demonstrates training and testing a machine learning model to classify handwritten digits using the MNIST dataset. The model leverages PyTorch for deep learning and is implemented in a Jupyter Notebook.

## Overview

The MNIST dataset is a benchmark dataset in machine learning, consisting of 60,000 training and 10,000 testing grayscale images of handwritten digits (0-9), each sized 28x28 pixels.

In this project, the model is trained to classify these digits with high accuracy, and predictions are made on test samples, showcasing the results with visualizations.

## Features
- Loading and preprocessing the MNIST dataset.
- Training a convolutional neural network (CNN) on the dataset.
- Visualizing sample predictions.
- Using PyTorch for deep learning.

## Requirements

Ensure the following Python packages are installed:

- Python 3.8+
- torch
- torchvision
- matplotlib

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mnist-digit-classification.git
   cd mnist-digit-classification
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook MNIST.ipynb
   ```

3. Run all cells to train the model and visualize predictions.

4. Modify the notebook to experiment with hyperparameters, model architecture, or additional visualizations.

## Key Sections in the Notebook

- **Data Loading**: Utilizes `torchvision` to fetch and preprocess the MNIST dataset.
- **Model Definition**: A simple yet effective CNN is implemented.
- **Training Loop**: The model is trained using a standard optimization loop.
- **Evaluation**: Accuracy is measured on the test set, and predictions are visualized for sample images.

## Example Output

The notebook displays the following:
- Sample test images with predicted labels.
- Training and test accuracy.

![Example Prediction](example_prediction.png)

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [NeuralNine](https://www.youtube.com/watch?v=vBlO87ZAiiw)
- The creators of the MNIST dataset
