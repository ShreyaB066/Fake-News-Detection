import streamlit as st
import joblib
import nltk
import re
from newspaper import Article
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

def predict_news_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        if len(text.strip()) == 0:
            return "Couldn't extract content."

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        return "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    except:
        return "❌ Error reading article."

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection from URL")

url_input = st.text_input("Paste News Article URL", placeholder="https://example.com/news")

if st.button("Check"):
    if url_input:
        with st.spinner("Analyzing article..."):
            result = predict_news_from_url(url_input)
        st.success(result)
    else:
        st.warning("Please enter a valid URL.")
