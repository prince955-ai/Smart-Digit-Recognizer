
# Smart Digit Recognizer 🔢

A Deep Learning project that recognizes handwritten digits
from 0 to 9 using a Convolutional Neural Network (CNN).

## Overview

Smart Digit Recognizer is an image classification application
built using Python, TensorFlow, and Keras. It uses the MNIST
dataset to learn patterns in handwritten digits and predict
the digit in an uploaded image.

## Features

- Deep learning with a CNN architecture
- MNIST handwritten digit dataset
- Image normalization and preprocessing
- Model training and evaluation
- Saved trained model
- Interactive Streamlit web application
- Prediction probabilities for digits 0-9

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- MNIST dataset

## CNN Architecture

1. Input layer: 28 x 28 grayscale image
2. Convolutional layer with 32 filters
3. Max pooling layer
4. Convolutional layer with 64 filters
5. Max pooling layer
6. Flatten layer
7. Dense layer with 128 neurons
8. Dropout layer
9. Output layer with 10 classes

## Installation

Clone this repository:

    git clone YOUR_GITHUB_REPOSITORY_URL

Move into the project directory:

    cd Smart-Digit-Recognizer

Install the dependencies:

    python -m pip install -r requirements.txt

## Train the Model

Run:

    python train_model.py

The script downloads the MNIST dataset, trains the CNN,
evaluates it, and saves the model in the models folder.

## Run the Web Application

After training, run:

    python -m streamlit run app.py

Open the local URL displayed in your terminal.
Upload a clear handwritten digit image and click
"Recognize Digit".

## Results

The training script reports test accuracy and test loss.
The actual values are saved in results.json.

Model performance may vary with training settings and
uploaded images. MNIST test accuracy does not guarantee
the same performance on every real-world photograph.

## Learning Outcomes

- Understanding convolutional neural networks
- Image classification using deep learning
- Training, validation, and testing
- Image preprocessing and normalization
- Model evaluation and deployment
- Building a simple ML web interface

## Future Improvements

- Add a drawing canvas
- Improve preprocessing for real-world images
- Compare CNN architectures
- Add confusion matrix and training graphs

## Author

Prince

BTech CSE (AI/ML)
