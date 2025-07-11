import streamlit as st
import joblib
import nltk
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean text
def clean_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

# Extract text from URL using BeautifulSoup
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return None

# Prediction logic
def predict_news_from_url(url):
    text = get_article_text(url)
    if not text or len(text.strip()) < 100:
        return "❌ Couldn't extract enough content. Try another link."

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "🟢 Real News" if prediction == 1 else "🔴 Fake News"

# Streamlit interface
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection from URL")
st.markdown("Paste the link of a news article below. We'll tell you if it's real or fake.")

url_input = st.text_input("Enter News Article URL", placeholder="https://example.com/article")

if st.button("Check"):
    if url_input:
        with st.spinner("Analyzing article..."):
            result = predict_news_from_url(url_input)
        st.success(result)
    else:
        st.warning("Please enter a URL.")
