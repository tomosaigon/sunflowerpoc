# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import unicodedata
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Define the text cleaning function
def clean_text(text):
  text = text.lower()   # Convert text to lower case
  text = re.sub(r'\[.*?\]', '', text)   # Remove text in square brackets
  text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)   # Remove punctuation
  text = re.sub(r'\w*\d\w*', '', text)   # Remove words containing numbers
  return text

# Load the unseen data
df_unseen = pd.read_csv('unseen_data.csv')

# Clean the 'content' column of unseen data
df_unseen['content'] = df_unseen['content'].apply(lambda x: clean_text(str(x)))

# Initialize the tokenizer with a specified vocabulary size
tokenizer = Tokenizer(num_words=5000)

# Fit the tokenizer on your text data
tokenizer.fit_on_texts(df_unseen['content'])

# Transform your text data to sequences of integers
unseen_sequences = tokenizer.texts_to_sequences(df_unseen['content'])

# Pad sequences so that they all have the same length
unseen_data_proc = pad_sequences(unseen_sequences, maxlen=400)

# Load the model
model = load_model('my_model.h5')

# Use the model to make predictions
predictions = model.predict(unseen_data_proc)

# Convert probabilities to class labels
predictions_labels = ['Community' if prob >= 0.5 else 'Crowd' for prob in predictions]

# Add the predictions as a new column in your DataFrame
df_unseen['predictions'] = predictions_labels

# Save the DataFrame with the predictions to a new CSV file
df_unseen.to_csv('unseen_data_with_predictions.csv', index=False)
