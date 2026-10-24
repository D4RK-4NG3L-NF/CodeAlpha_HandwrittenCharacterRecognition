# Handwritten Character Recognition

## Project Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

## Technologies Used
- Python
- TensorFlow and Keras
- MNIST dataset
- VS Code

## Project Files
- `prepare_data.py` — loads and prepares the dataset.
- `train_model.py` — builds, trains, evaluates, and saves the CNN model.
- `predict_digit.py` — predicts a digit from a test image.
- `handwritter_digit_model.keras` — the saved trained model.

## Results
- Test accuracy: **98.05%**
- Sample prediction: predicted digit **7**, correct digit **7**.

## How to Run
1. Install the required Python packages.
2. Run `python prepare_data.py`.
3. Run `python train_model.py` to train the model.
4. Run `python predict_digit.py` to test a prediction.

## Purpose
This beginner AI/ML project demonstrates image preprocessing, CNN training, model evaluation, and digit prediction.
