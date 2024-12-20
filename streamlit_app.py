import os
import pinecone
import streamlit as st
from langchain.llms import HuggingFaceHub
from huggingface_hub import login
import PyPDF2

pinecone_api_key = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

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
