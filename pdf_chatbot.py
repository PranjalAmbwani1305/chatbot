import pinecone
import pdfplumber
from sentence_transformers import SentenceTransformer
import os

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index = pinecone.Index("pdf-chatbot")

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_embeddings(text):
    return model.encode([text])[0]

def store_in_pinecone(text, doc_id):
    embeddings = get_embeddings(text)
    index.upsert([(doc_id, embeddings)])

def query_pinecone(query):
    query_embedding = get_embeddings(query)
    result = index.query([query_embedding], top_k=1, include_metadata=True)
    return result['matches'][0]['metadata']['text'] if result['matches'] else "No relevant information found."
