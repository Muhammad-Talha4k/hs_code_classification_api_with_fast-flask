'''from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model 
import re
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the ANN model, vectorizer, and label encoder
model = load_model('ANN_model.h5')  # Load the ANN model using Keras
vectorizer = joblib.load('ann_vectorizer.pkl')
label_encoder = joblib.load('ann_label_encoder.pkl')

# Preprocessing function
def preprocess(description):
    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase
    text = description.lower()

    # Remove special characters
    text = re.sub(r'[^a-z0-9 ]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a string
    cleaned_description = ' '.join(tokens)

    return cleaned_description

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess each description
    preprocessed_descriptions = [preprocess(desc) for desc in data['descriptions']]

    # Transform the descriptions using the vectorizer
    vectorized_descriptions = vectorizer.transform(preprocessed_descriptions)

    # Convert the sparse matrix to a dense matrix if necessary
    vectorized_descriptions_dense = vectorized_descriptions.toarray()

    # Make predictions using the trained ANN model
    predictions_prob = model.predict(vectorized_descriptions_dense)
    
    # Get the top 3 class indices and probabilities
    top_n = 3
    top_indices = np.argsort(predictions_prob, axis=1)[:, -top_n:][:, ::-1]
    top_probs = np.sort(predictions_prob, axis=1)[:, -top_n:][:, ::-1]

    # Decode the top class indices to class labels
    top_labels = label_encoder.inverse_transform(top_indices.flatten()).reshape(top_indices.shape)

    # Create a list of dictionaries with descriptions and their corresponding top 3 predictions
    result = []
    for i, description in enumerate(data['descriptions']):
        predictions = []
        for j in range(top_n):
            predictions.append({
                "commodity_code": top_labels[i][j],
                "probability": f"{int(top_probs[i][j] * 100)}%" 
            })
        result.append({
            "description": description,
            "predictions": predictions
        })

    # Return predictions as JSON
    return jsonify({"predictions": result})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)'''

from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the model, vectorizer, and label encoder
model = joblib.load('sgd.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Preprocessing function
def preprocess(description):
    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase
    text = description.lower()

    # Remove special characters
    text = re.sub(r'[^a-z0-9 ]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a string
    cleaned_description = ' '.join(tokens)

    return cleaned_description

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess each description
    preprocessed_descriptions = [preprocess(desc) for desc in data['descriptions']]

    # Transform the descriptions using the vectorizer
    vectorized_descriptions = vectorizer.transform(preprocessed_descriptions)

    # Get the probabilities for each class
    probabilities = model.predict_proba(vectorized_descriptions)

    # Get the top 3 class indices and probabilities
    top_n = 3
    top_indices = np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]
    top_probs = np.sort(probabilities, axis=1)[:, -top_n:][:, ::-1]

    # Decode the top class indices to class labels
    top_labels = label_encoder.inverse_transform(top_indices.flatten()).reshape(top_indices.shape)

    # Create a list of dictionaries with descriptions and their corresponding top 3 predictions
    result = []
    for i, description in enumerate(data['descriptions']):
        result.append({
            "description": description,
            "predictions": [
                {"commodity_code": top_labels[i][j], "probability": f"{int(top_probs[i][j] * 100)}%" }
                for j in range(top_n)
            ]
        })

    # Return predictions as JSON
    return jsonify({"predictions": result})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)