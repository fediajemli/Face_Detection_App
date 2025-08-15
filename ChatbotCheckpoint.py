# app.py
import streamlit as st
import re
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# ----------------- NLTK Stopwords Setup -----------------
import nltk

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


# ----------------- Preprocessing Function -----------------
def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Split words manually and remove stopwords
    preprocessed_sentences = []
    for sent in sentences:
        words = sent.split()  # simple word splitting
        filtered = [w for w in words if w not in stop_words]
        preprocessed_sentences.append(' '.join(filtered))

    return preprocessed_sentences


# ----------------- Similarity Function -----------------
def get_most_relevant_sentence(user_input, sentences):
    sentences.append(user_input)  # Add user input for comparison
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = np.argmax(similarity_scores)
    sentences.pop()  # Remove user input
    return sentences[index]


# ----------------- Chatbot Function -----------------
def chatbot(user_input, sentences):
    response = get_most_relevant_sentence(user_input, sentences)
    return response


# ----------------- Streamlit Interface -----------------
def main():
    st.title("📚 Text-based Chatbot")
    st.markdown(
        "This chatbot answers questions based on the text in **pg236.txt**. Ask anything related to the content of this file!")

    # Load and preprocess text file
    if not os.path.exists("pg236.txt"):
        st.error("File 'pg236.txt' not found in the project folder.")
        return

    with open("pg236.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentences = preprocess(text)

    # User input
    user_input = st.text_input("Ask a question:")

    if user_input:
        response = chatbot(user_input, sentences)
        st.write("🤖 Chatbot:", response)


if __name__ == "__main__":
    main()

