import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import string
from transformers import pipeline
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load datasets
try:
    input_table_path = "input_table.csv"
    qa_dataset_path = "QA_dataset_share.xlsx"
    input_table = pd.read_csv(input_table_path)
    qa_dataset = pd.read_excel(qa_dataset_path)
    logging.info("Datasets loaded successfully.")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    input_table = None
    qa_dataset = None

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Initialize NLP models
try:
    qa_pipeline = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-squad2")
    query_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logging.info("NLP models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading NLP models: {e}")
    qa_pipeline = None
    query_classifier = None

# Query classification function
def classify_query(query):
    """Classify query type into text-based, numerical, or filtering-based."""
    categories = ["text", "numerical", "filter"]
    result = query_classifier(query, categories)
    return result['labels'][0] if result['scores'][0] > 0.5 else "text"

# Retrieve relevant data using TF-IDF similarity
def retrieve_relevant_data(query, table):
    try:
        if table is None or table.empty:
            logging.error("Table data is empty or not loaded.")
            return None
        corpus = table.apply(lambda row: ' '.join(map(str, row)), axis=1)
        vectorizer = TfidfVectorizer().fit_transform([query] + corpus.tolist())
        cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
        top_indices = cosine_sim.argsort()[-5:][::-1]  # Retrieve top 5 relevant rows
        return table.iloc[top_indices]
    except Exception as e:
        logging.error(f"Error in retrieve_relevant_data: {e}")
        return None

# Structured query execution (filtering, counting)
def execute_structured_query(query, table):
    try:
        if "transactions" in query.lower() and "male" in query.lower() and "rating 9.1" in query.lower():
            return str(table[(table["Gender"] == "Male") & (table["Rating"] == 9.1)].shape[0])
        return None
    except Exception as e:
        logging.error(f"Error in execute_structured_query: {e}")
        return None

# Generate final answer
def generate_final_answer(query, relevant_data, query_type):
    try:
        if relevant_data is None or relevant_data.empty:
            return "No relevant data found."
        
        if query_type == "filter":
            structured_answer = execute_structured_query(query, input_table)
            if structured_answer:
                return structured_answer
        
        context = ' '.join([f"{col}: {val}" for _, row in relevant_data.iterrows() for col, val in row.items()])
        if qa_pipeline:
            response = qa_pipeline(question=query, context=context)
            return response['answer'] if response['score'] > 0.3 else "No confident answer found."
        
        return "No confident answer found."
    except Exception as e:
        logging.error(f"Error in generate_final_answer: {e}")
        return "Error generating response."

# Calculate evaluation metrics
def calculate_evaluation_metrics(predicted_answer, query):
    try:
        ground_truth_row = qa_dataset[qa_dataset['question'].str.lower() == query.lower()]
        if ground_truth_row.empty:
            return 0, 0  # No matching ground truth found
        true_answer = ground_truth_row['answer'].values[0]
        
        f1 = f1_score([true_answer], [predicted_answer], average='micro')
        accuracy = accuracy_score([true_answer], [predicted_answer])
        return round(f1, 2), round(accuracy, 2)
    except Exception as e:
        logging.error(f"Error in calculating F1 Score and Accuracy: {e}")
        return 0, 0

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('question', '')
        if not query:
            return jsonify({'final_answer': "No question provided.", 'f1_score': 0, 'accuracy': 0})
        
        query_type = classify_query(query)
        relevant_data = retrieve_relevant_data(query, input_table)
        if relevant_data is None:
            return jsonify({'final_answer': "No relevant data found.", 'f1_score': 0, 'accuracy': 0})
        
        response = generate_final_answer(query, relevant_data, query_type)
        f1, accuracy = calculate_evaluation_metrics(response, query)
        
        return jsonify({
            'final_answer': response,
            'f1_score': f1,
            'accuracy': accuracy
        })
    except Exception as e:
        logging.error(f"Error in search API: {e}")
        return jsonify({'final_answer': "Server error occurred.", 'f1_score': 0, 'accuracy': 0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
