import streamlit as st
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

# For environment/secrets support
api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

@st.cache_resource(show_spinner=False)
def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions, answers = [], []
    for item in data:
        q, a = item.get('q'), item.get('Response')
        if q and a:
            questions.append(q.lower())
            answers.append(a)
    return questions, answers

faq_questions, faq_answers = load_dataset("data/train.json")
vectorizer = TfidfVectorizer(stop_words='english')
faq_vectors = vectorizer.fit_transform(faq_questions)

def retrieve_context(user_query):
    query_vec = vectorizer.transform([user_query.lower()])
    sims = cosine_similarity(query_vec, faq_vectors)
    idx = sims.argmax()
    score = sims[0][idx]
    if score < 0.1:
        return None, None
    return faq_questions[idx], faq_answers[idx]

def get_gemini_response(user_query):
    context_q, context_a = retrieve_context(user_query)
    if context_q and context_a:
        prompt = (
            "You are an expert Indian legal assistant AI. Use the following reference Q&A to help answer the query:\n\n"
            f"Reference Question: {context_q}\nReference Answer: {context_a}\n\n"
        )
    else:
        prompt = "You are an expert Indian legal assistant AI. Answer clearly and precisely:\n\n"
    prompt += f"User's question: {user_query}\nAI's answer:"
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Rock Smashers | Indian Legal Q&A", layout="wide")
st.markdown('<h1 style="color:#00ffe7;text-align:center;font-family:Orbitron,sans-serif;">Rock Smashers<br/><span style="font-size:1rem;color:#0ff;">Indian Legal Q&A</span></h1>', unsafe_allow_html=True)
st.write("")

user_query = st.text_area("Ask your legal question here:", height=120)
if st.button("🧬 Analyze & Respond"):
    if user_query.strip():
        st.markdown(f'<div style="font-size:1.1rem;color:#00ffd2;font-weight:700;margin-top:42px;">You asked:</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#001319cc;border-left:5px solid #00ffe7;padding:16px 22px;font-size:1.3rem;border-radius:0 14px 14px 0;margin-bottom:18px;color:#fff;">{user_query}</div>', unsafe_allow_html=True)
        answer = get_gemini_response(user_query)
        st.markdown(f'<div style="font-size:1.17rem;color:#00ffe7;"><strong>🤖 Legal AI:</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:rgba(0,255,231,0.09);border-left:5px solid #00ffe7;padding:16px 22px;font-size:1.15rem;border-radius:0 14px 14px 0;margin-bottom:18px;color:#e0fff9;">{answer}</div>', unsafe_allow_html=True)
    else:
        st.error("Please enter a question.")

st.markdown("""
<hr style="border: 1.5px dashed #0af; margin: 36px 0;">
<p style="text-align:center;color:#00ffcaa1;font-size:0.99rem;">
Information is for reference only.<br>
No AI system can substitute certified legal counsel.<br>
© Rock Smashers, 2025
</p>
""", unsafe_allow_html=True)
