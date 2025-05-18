import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load FAQ data
with open("data/faq.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Preprocessing
stop_words = set(stopwords.words("english"))

def preprocess(text):
    return " ".join([word for word in text.lower().split() if word not in stop_words])

questions_cleaned = [preprocess(q) for q in questions]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions_cleaned)

# Streamlit UI
st.title("FAQ Chatbot 🤖")
user_input = st.text_input("Ask a question:")

if user_input:
    user_cleaned = preprocess(user_input)
    user_vec = vectorizer.transform([user_cleaned])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()

    if similarity[0][index] > 0.2:
        st.success(answers[index])
    else:
        st.error("Sorry, I couldn't understand your question. Please rephrase.")
