'''from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
import numpy as np
import uvicorn

# Initialize the FastAPI application
app = FastAPI()

# Load the trained logistic regression model, label encoder, and vectorizer
model = joblib.load('lr_bow.pkl')  # Load logistic regression model
vectorizer = joblib.load('count_vectorizer.pkl')  # Load vectorizer
label_encoder = joblib.load('label_encoder.pkl')  # Load label encoder

# Define the request body structure
class DescriptionRequest(BaseModel):
    descriptions: list[str]

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
@app.post("/predict")
async def predict(request: DescriptionRequest):
    # Preprocess each description
    preprocessed_descriptions = [preprocess(desc) for desc in request.descriptions]

    # Transform the descriptions using the vectorizer
    vectorized_descriptions = vectorizer.transform(preprocessed_descriptions)

    # Make predictions using the trained logistic regression model
    predictions_prob = model.predict_proba(vectorized_descriptions)

    # Get the top 3 class indices and probabilities
    top_n = 3
    top_indices = np.argsort(predictions_prob, axis=1)[:, -top_n:][:, ::-1]
    top_probs = np.sort(predictions_prob, axis=1)[:, -top_n:][:, ::-1]

    # Decode the top class indices to class labels
    top_labels = label_encoder.inverse_transform(top_indices.flatten()).reshape(top_indices.shape)

    # Create a list of dictionaries with descriptions and their corresponding top 3 predictions
    result = []
    for i, description in enumerate(request.descriptions):
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
    return {"predictions": result}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''

from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import re
import joblib
import numpy as np
import uvicorn

# Initialize the FastAPI application
app = FastAPI()

# Load the ANN model, vectorizer, and label encoder
model = load_model('ANN_model.h5')  # Load the ANN model using Keras
vectorizer = joblib.load('ann_vectorizer.pkl')
label_encoder = joblib.load('ann_label_encoder.pkl')

# Define the request body structure
class DescriptionRequest(BaseModel):
    descriptions: list[str]

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
@app.post("/predict")
def predict(request: DescriptionRequest):

    # Preprocess each description
    preprocessed_descriptions = [preprocess(desc) for desc in request.descriptions]

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
    for i, description in enumerate(request.descriptions):
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
    return {"predictions": result}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
