import os
import json
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- NLTK Download: Download stopwords once, preferably outside the app's runtime ---
# It's best practice to run this once from your terminal: python -c "import nltk; nltk.download('stopwords')"
# Or, if you must keep it in the script for first-time setup, ensure it's not run on every app restart
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("NLTK stopwords not found, downloading now...")
    nltk.download('stopwords')
# --- End NLTK Download ---

# Configure Gemini API using the loaded API key
# Ensure GEMINI_API_KEY is set in your .env file (e.g., GEMINI_API_KEY="your_actual_api_key")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    # You might want to raise an error or exit here in a production environment
    # For development, we'll proceed but the Gemini API calls will fail without a valid key.

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# Define the model to use
model = genai.GenerativeModel('gemini-1.5-flash') # Change model if needed

def clean_text(text):
    """Lowercase, remove special characters except spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset(filepath):
    """Load JSON dataset with keys 'q' and 'Response'."""
    questions = []
    answers = []
    if not os.path.exists(filepath):
        print(f"Error: Dataset file not found at {filepath}")
        return questions, answers
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                q = item.get('q')
                a = item.get('Response')
                if q and a:
                    questions.append(clean_text(q))
                    answers.append(a)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred loading dataset from {filepath}: {e}")
    return questions, answers

# Load your dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'train.json') # Use absolute path
faq_questions, faq_answers = load_dataset(DATA_PATH)

# Initialize TF-IDF vectorizer and transform questions
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

# Only fit and transform if there are questions loaded
if faq_questions:
    faq_vectors = vectorizer.fit_transform(faq_questions)
else:
    print("Warning: No FAQ questions loaded. Context retrieval will not work.")
    faq_vectors = None # Set to None if no questions are loaded

def retrieve_context(user_query):
    """Retrieve the most relevant Q&A pair from dataset based on user query."""
    if faq_vectors is None or faq_vectors.shape[0] == 0:
        return None, None # No context to retrieve if dataset is empty

    query_clean = clean_text(user_query)
    query_vec = vectorizer.transform([query_clean])
    similarity = cosine_similarity(query_vec, faq_vectors)
    idx = similarity.argmax()
    score = similarity[0][idx]
    # Optionally set a minimal similarity threshold (e.g. 0.1)
    if score < 0.1: # This threshold might need tuning based on your data
        return None, None
    return faq_questions[idx], faq_answers[idx]

def get_gemini_response(user_query):
    """Generate response from Gemini API with dataset context augmented."""
    context_q, context_a = retrieve_context(user_query)
    
    # Construct prompt based on whether context was found
    if context_q and context_a:
        prompt = (
            "You are an expert Indian legal assistant AI. Use the below reference Q&A to answer the user query clearly:\n\n"
            f"Reference Q: {context_q}\nReference A: {context_a}\n\n"
        )
    else:
        prompt = (
            "You are an expert Indian legal assistant AI. Answer the user's query clearly and precisely:\n\n"
        )
    
    prompt += f"User's question: {user_query}\nAnswer:"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating content from Gemini API: {e}")
        return "I'm sorry, I couldn't process your request at the moment. Please try again later."


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        user_query = request.form.get('query', '')
        if user_query.strip():
            answer = get_gemini_response(user_query)
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    # Create the 'data' directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # You might want to create a dummy train.json for testing if it doesn't exist
    dummy_data_path = os.path.join(data_dir, 'train.json')
    if not os.path.exists(dummy_data_path):
        print(f"Creating a dummy {dummy_data_path} file.")
        with open(dummy_data_path, 'w', encoding='utf-8') as f:
            json.dump([
                {"q": "What is the capital of India?", "Response": "The capital of India is New Delhi."},
                {"q": "What is the highest court in India?", "Response": "The Supreme Court of India is the highest judicial court."}
            ], f, indent=4)

    app.run(debug=True)
