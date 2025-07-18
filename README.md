ML SMS Spam Classification

A machine learning project using deep learning (LSTM with word embeddings) to classify SMS messages as spam or not spam. Built with TensorFlow/Keras and trained on the SMS Spam Collection dataset, this model achieves 98% validation accuracy.

## Features:

Tokenization with NLTK

Custom word-to-index mapping

Padded sequences for variable-length messages

Keras Embedding + LSTM architecture

98% validation accuracy on the classic SMS spam dataset

Prediction on unseen test messages

Model Architecture:

Embedding layer: Converts words to dense vectors (32D)

LSTM layer: Learns temporal patterns in messages

Dense layer: Final binary classification using sigmoid activation

## Project Structure:

Classifier.py: Core training and prediction logic

SMSSpamCollection: Dataset (label + message)

README.md: Project overview and instructions

## Dataset:

5,574 SMS messages

Each message is labeled as either ham (not spam) or spam

The first 5,014 messages are used for training/validation

A few (6) messages are reserved for testing

## Installation and Setup:

Clone the repository:
git clone https://github.com/KatoTheFluffyWolf/ML-SMS-Spam-Classification.git
cd ML-SMS-Spam-Classification

Install required libraries:
pip install tensorflow nltk numpy

Run the training script:
python Classifier.py

## Evaluation:

Training/Validation Split: 80% / 20%

Validation Accuracy: ~98%

Sample output on test messages:
Message: Congratulations! You've won a $1000 Walmart gift card.
Actual Label: Spam
Predicted Probability: 0.92
Predicted Label: Spam

## Prediction:
After training, the script prints predictions on 5 unseen test messages. You can modify Classifier.py to input your own messages for live predictions.

## Dependencies:

Python 3.x

TensorFlow

NLTK

NumPy

## Contributing:
Contributions are welcome. Please:

Fork the repository

Create a new feature branch

Submit a pull request with a detailed description

## License:
This project is licensed under the MIT License.

Author:
Duy Anh Nguyen
GitHub: https://github.com/KatoTheFluffyWolf

[README.md](https://github.com/user-attachments/files/21308310/README.md)
