import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

file_path = 'norm.txt'
norm = {}
with open(file_path, 'r') as f:
    for line in f:
        key, value = line.strip().split(':', 1)  # Pisahkan key dan value
        norm[key.strip().replace('"', '')] = value.strip().replace('"', '')

def remove_duplicates(df):
  df_cleaned = df.drop_duplicates(subset=['full_text'], keep='first')
  return df_cleaned

def clean_twitter_text(text):
  text = re.sub(r'http\S+', '', text)
  text = re.sub(r'pic.twitter.com\S+', '', text)
  text = re.sub(r'@[A-Za-z0-9_]+', '', text)
  text = re.sub(r'#\w+', '', text)
  text = re.sub(r'RT[\s]+', '', text)
  text = re.sub(r'https?://\S+', '', text)
  text = re.sub(r'\d+', '', text)
  text = re.sub(r'-', ' ', text)
  text = re.sub(r'[^A-Za-z0-9 ]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text.lower()

def bersihkan_promosi(teks):
  pola = r"download.*$"
  teks_bersih = re.sub(pola, "", teks, flags=re.IGNORECASE)
  return teks_bersih.strip()


def normalize_text(text):
    words = text.split()
    normalized_words = [norm[word] if word in norm else word for word in words]
    normalized_text = ' '.join(normalized_words)
    return normalized_text.lower()

exclude_words = ["menangis","setuju", "naturalisasi", "indonesia", "mendukung", "dukung", "dukungan", "persetujuan", "WNI", "warga negara", "asing", "proses", "pemain", "timnas", "tidak", "bukan", "jangan"]
def remove_stopwords(text, exclude_words=[]):
       # Handle float values
       if isinstance(text, float):
           text = str(text)  # Convert float to string
       tokens = word_tokenize(text.lower())
       filtered_tokens = [token for token in tokens if token not in stopwords.words('indonesian') or token in exclude_words]
       processed_text = ' '.join(filtered_tokens)
       return processed_text

def stemming(text, exclude_words=[]):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  words = word_tokenize(text)
  stemmed_words = [word if word in exclude_words else stemmer.stem(word) for word in words]
  stemmed_text = " ".join(stemmed_words)
  return stemmed_text
def outlier_delete(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df['full_text_length'].quantile(0.25)
    Q3 = df['full_text_length'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df['full_text_length'] >= lower_bound) & (df['full_text_length'] <= upper_bound)]
    
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df
    
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.dropna()
    df_cleaned = outlier_delete(df_cleaned)
    
    # Apply text cleaning functions to 'full_text' column
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(bersihkan_promosi)
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(clean_twitter_text)
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(normalize_text)
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(lambda x: remove_stopwords(x, exclude_words))
    df_cleaned['full_text'] = df_cleaned['full_text'].apply(lambda x: stemming(x, exclude_words))
    
    df_cleaned = remove_duplicates(df_cleaned)
    df_cleaned = df_cleaned.dropna()
    
    # Remove extra spaces in 'full_text'
    df_cleaned['full_text'] = df_cleaned['full_text'].str.replace('  ', ' ', regex=False)
    df_cleaned['full_text'] = df_cleaned['full_text'].str.strip()
    
    return df_cleaned

if __name__ == "__main__":
  print("Starting preprocessing of naturaisasi dataset...")
  df = pd.read_csv('naturalisasi_dataset.csv')
  df['full_text_length'] = df['full_text'].apply(len)
  df = preprocess_text(df)
    
  df_cleaned_path = 'preprocessing/naturalisasi_dataset_cleaned.csv'
  df.to_csv(df_cleaned_path, index=False)
  print(f"Processed dataset saved to {df_cleaned_path}")
