## HS Code Classification API's with FastAPI and Flask


## Description

This repository contains a HS code classification API'S built using FastAPI and Flask. The API provides endpoints for making predictions using trained machine learning models of logistic regression and artificial neural network.

## Project Structure

- `ANN_model.h5`: Trained Artificial Neural Network (ANN) model.
- `ann_label_encoder.pkl`: Label encoder for ANN model.
- `ann_vectorizer.pkl`: Vectorizer for ANN model.
- `count_vectorizer.pkl`: Bag of words vectorizer.
- `label_encoder.pkl`: General label encoder.
- `lr_bow.pkl`: Trained logistic regression model with bag of words.
- `main.py`: FastAPI implementation.
- `main_flask.py`: Flask implementation.

### Workflow of API
- Client sends a POST request to the /predict endpoint with a JSON body containing descriptions.
- FastAPI receives the request and passes the descriptions to the predict function.
- Descriptions are preprocessed using the preprocess function, which:
  1. Converts the text to lowercase.
  2. Removes special characters.
  3. Tokenizes the text.
  4. Removes stop words.
  5. Lemmatizes the tokens.
  6. Joins the tokens back into a cleaned string.
- Preprocessed descriptions are transformed using the vectorizer.
- Model makes predictions based on the transformed descriptions and provides probabilities for each class.
- From the predicted probabilities, the top 3 class indices and their respective probabilities are extracted.
- Predictions are decoded using the label encoder.
- A response, containing the original descriptions and their corresponding top 3 predictions, with each prediction including the commodity code and its probability.
- Predictions are returned as a JSON response to the client.

### Installation


### Requirements

- Python 3.7+
- FastAPI
- Flask
- scikit-learn
- TensorFlow

### Setup 

 Clone the repository:
 
 ```bash git clone https://github.com/Muhammad-Talha4k/machine_learning_api_with_fast-flask.git cd machine_learning_api_with_fast-flask ```
 Supervised Learning used
 
### Usage

You can make predictions by sending a POST request to the appropriate endpoint with input descriptions in a json format .

### Contributions
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
