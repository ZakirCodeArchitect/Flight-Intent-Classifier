# ATIS Intent Classification Model

This repository contains an ATIS (Airline Travel Information System) intent classification model built using deep learning techniques. The model is designed to classify user queries related to flight schedules, airfare, flight details, and more, into predefined intents such as `atis_flight`, `atis_airfare`, and `atis_flight_time`.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Usage](#usage)
- [Installation](#installation)
- [License](#license)

## Project Overview

This project uses natural language processing (NLP) techniques and deep learning to classify flight-related queries into specific intents. The model is trained using the ATIS dataset and employs a Bidirectional LSTM network for intent classification.

### Key Features:
- **Intent Classification:** Classify user queries into one of 8 intents.
- **Accuracy:** Achieves a test accuracy of **97%** and high F1-scores for most intents.
- **Model:** Uses an embedding layer followed by two bidirectional LSTM layers and dense layers for classification.

## Dataset

The model is trained on the [ATIS dataset](https://www.kaggle.com/datasets/uciml/atis), which consists of natural language queries related to flight details. The dataset is divided into:
- **Training Data:** 4,833 queries
- **Testing Data:** 799 queries

### Data Columns:
- **Intent:** The intent label (e.g., `atis_flight`, `atis_airfare`).
- **Query:** The user's flight-related query.

## Model Architecture

The model architecture consists of the following layers:
1. **Embedding Layer:** Converts words into 128-dimensional vectors.
2. **Bidirectional LSTM Layers:** Capture contextual information from both directions of the input sequence.
3. **Dropout Layers:** Help prevent overfitting by randomly dropping connections during training.
4. **Dense Layers:** Final classification layer that outputs the predicted intent.

### Hyperparameters:
- **Vocabulary Size:** 872
- **Max Sequence Length:** 46
- **Number of Classes (Intents):** 8
- **Total Parameters:** 253,992 (All trainable)

## Training

- The model is trained for 10 epochs.
- **Training Accuracy:** Ranges from 96.25% (Epoch 1) to 98.77% (Epoch 10).
- **Validation Accuracy:** Peaked at 98.24%.
- **Loss:** Training loss decreased from 0.1218 to 0.0457, with minimal overfitting.

### Observations:
- The model shows excellent performance, but certain underrepresented classes such as `atis_quantity` and `atis_flight_time` need more attention.
- **Training Accuracy:** 98.77%
- **Validation Accuracy:** 98.24%

## Testing and Evaluation

- **Test Accuracy:** 97%
- **Precision, Recall, and F1-Score:** 
  - `atis_flight`: Precision 0.99, Recall 0.98
  - `atis_airfare`: Precision 0.90, Recall 0.90
  - `atis_abbreviation`: Precision 0.94, Recall 1.00
  - F1-Scores are lower for some classes like `atis_quantity` (0.55) and `atis_flight_time` (0.67).

## Usage

To classify the intent of a query, use the `predict_intent()` function. Example:

```python
from intent_classifier import predict_intent

query = "show me flights from New York to Los Angeles"
intent = predict_intent(query)
print(f"Predicted Intent: {intent}")
```
