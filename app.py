import streamlit as st
import joblib
import nltk
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean and preprocess text
def clean_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

# Extract article text using requests + BeautifulSoup
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs[:10]])  # limit to first 10
        return text
    except Exception as e:
        return None

# Predict real or fake
def predict_news_from_url(url):
    text = get_article_text(url)
    if not text or len(text.strip()) < 100:
        return "❌ Couldn't extract enough content. Try a different URL."
    
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0][prediction]
    label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    return f"{label} (Confidence: {confidence:.2f})"

# Streamlit web interface
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection from URL")
st.caption("⚠️ This is a basic ML model trained on limited data. Results may not be fully accurate.")

url_input = st.text_input("Enter News Article URL", placeholder="https://example.com/article")

if st.button("Check"):
    if url_input:
        with st.spinner("Analyzing..."):
            result = predict_news_from_url(url_input)
        st.success(result)
    else:
        st.warning("Please enter a valid URL.")

