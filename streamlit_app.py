import os
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
from huggingface_hub import login
import streamlit as st
from huggingface_hub import login
import PyPDF2

os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot using Hugging Face, Pinecone, and Langchain")

huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "your-huggingface-api-token")
login(token=huggingface_api_token)

model = HuggingFaceHub(
    repo_id="microsoft/layoutlmv2-base-uncased",
    huggingfacehub_api_token=huggingface_api_token,
    task="text-generation"
)

def get_response(user_input):
    response = model(user_input)
    return response

def store_conversation(user_input, response):
    index_name = "chatbot_conversations"
    pinecone_index = pinecone.Index(index_name)
    pinecone_index.upsert([(user_input, response)])

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
user_input = st.text_input("You: ", "")

if pdf_file:
    pdf_text = read_pdf(pdf_file)
    st.write(f"PDF Content: {pdf_text}")
    response = get_response(pdf_text)
    store_conversation(pdf_text, response)
    st.write(f"Chatbot: {response}")
elif user_input:
    response = get_response(user_input)
    store_conversation(user_input, response)
    st.write(f"Chatbot: {response}")
