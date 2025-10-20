from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure Gemini
configure(api_key=GOOGLE_API_KEY)

# Initialize HuggingFace embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

model = GenerativeModel("gemini-2.5-flash")

def gemini_rag_query(question):
    # Retrieve relevant context
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an assistant for medical question-answering tasks.
    Use the following context to answer the user's question.
    If you don't know the answer, say "I don't know".
    Keep the answer concise and within five sentences.

    Context:
    {context}

    Question: {question}
    Answer:
    """

    # Get response directly from Gemini
    response = model.generate_content(prompt)
    return response.text.strip()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)
    
    response = gemini_rag_query(msg)
    print("Gemini Response:", response)
    
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
