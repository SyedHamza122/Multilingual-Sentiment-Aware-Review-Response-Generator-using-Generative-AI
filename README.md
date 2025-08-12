# Multilingual-Sentiment-Aware-Review-Response-Generator-using-Generative-AI

!pip install --quiet kaggle pandas numpy tqdm langdetect sentence-transformers "bertopic[all]" hdbscan umap-learn transformers torch accelerate tokenizers sentencepiece matplotlib wordcloud scikit-learn
///
from google.colab import files
uploaded = files.upload()  # choose your kaggle.json locally
import os
os.makedirs('/root/.kaggle', exist_ok=True)
for f in uploaded:
    os.rename(f, '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
print("kaggle.json uploaded & configured.")
///
# For Amazon Reviews Multi:
!kaggle datasets download -d mexwell/amazon-reviews-multi -p /content/data --unzip

# Or for Google Play Store Reviews:
!kaggle datasets download -d prakharrathi25/google-play-store-reviews -p /content/data --unzip
///
!ls -la /content/data
///
import pandas as pd
import glob

# List all CSVs from /content/data
csv_files = glob.glob('/content/data/**/*.csv', recursive=True) + glob.glob('/content/data/*.csv')
print("Found CSV files:", csv_files)

# Load first CSV
df = pd.read_csv(csv_files[0], low_memory=False)
print("Shape:", df.shape)
df.head()
///
SAMPLE_SIZE = 500
df_sample = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
df_sample.head()
///
from langdetect import detect
from tqdm import tqdm

tqdm.pandas()

def safe_detect(text):
    try:
        return detect(text)
    except:
        return "unknown"

df_sample['detected_lang'] = df_sample['review_body'].progress_apply(safe_detect)
df_sample.head()
///
!pip install -q bertopic[all] sentence-transformers
!pip install -q transformers
///
!pip install -q transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load multilingual sentiment model
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ['negative', 'neutral', 'positive']

def predict_sentiment(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = output.logits[0].detach().numpy()
    sentiment = labels[np.argmax(scores)]
    return sentiment

# Choose text column (translated if exists)
text_column = 'translated_review' if 'translated_review' in df_sample.columns else 'review_body'

# Apply sentiment prediction
df_sample['sentiment'] = df_sample[text_column].apply(predict_sentiment)

# Show sample output
df_sample[['review_body', 'topic', 'sentiment']].head()
///
!pip install -q matplotlib wordcloud

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1️⃣ Sentiment Distribution Pie Chart
sentiment_counts = df_sample['sentiment'].value_counts()

plt.figure(figsize=(5,5))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.show()

# 2️⃣ Word Cloud per Sentiment
for sentiment in df_sample['sentiment'].unique():
    text_data = " ".join(df_sample[df_sample['sentiment'] == sentiment]['review_body'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'Word Cloud - {sentiment.capitalize()} Reviews')
    plt.show()
///
