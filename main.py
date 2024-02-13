from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
import pandas as pd
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel
from typing import List
import tempfile
import re
import string
import pickle   # for saving and loading the tokenizer

# Define the text cleaning function
def clean_text(text):
      text = text.lower()   # Convert text to lower case
      text = re.sub(r'\[.*?\]', '', text)   # Remove text in square brackets
      text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)   # Remove punctuation
      text = re.sub(r'\w*\d\w*', '', text)   # Remove words containing numbers
      return text

app = FastAPI()

# Load the model and tokenizer when the app starts
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('my_model.h5')

class Item(BaseModel):
    content: str

@app.post("/upload/")
async def upload_file(content: str = Form(...)):
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
        temp_file.write(content)

    # df_unseen = pd.read_csv('unseen_data.csv')
    df_unseen = pd.read_csv(temp_file.name)

    # Clean the 'content' column of unseen data
    df_unseen['content'] = df_unseen['content'].apply(lambda x: clean_text(str(x)))

    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
          tokenizer = pickle.load(handle)
    
    # Transform your text data to sequences of integers
    unseen_sequences = tokenizer.texts_to_sequences(df_unseen['content'])
    # Pad sequences so that they all have the same length
    unseen_data_proc = pad_sequences(unseen_sequences, maxlen=400)

    # Load the model
    model = load_model('my_model.h5')
    # Make a prediction
    predictions = model.predict(unseen_data_proc)

    # Convert probabilities to class labels
    predictions_labels = ['Community' if prob >= 0.5 else 'Crowd' for prob in predictions]

    # Return the prediction
    return {'prediction': predictions_labels}

@app.get("/")
def form_post(request: Request):
    return HTMLResponse("""
        <html>
            <body>
                <form action="/upload" method="post">
                    <textarea name="content" rows="4" cols="50"></textarea><br>
                    <input type="submit" value="Predict">
                </form>
            </body>
        </html>
    """)