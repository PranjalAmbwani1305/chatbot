import os
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import login

# Log in to Hugging Face (use your actual token)
login(token='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') #Replace with your token

# Load environment variables
load_dotenv()

# Set up API keys
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class CustomChatbot:
    # ... (rest of the CustomChatbot class remains the same)

def clean_response_string(text):
    if isinstance(text, str):
        return text.replace("\uf8e7", "").replace("\xad", "").replace("\\n", "\n").replace("\t", " ")
    return "Not a string"

def extract_and_clean_text_from_dict(data, key_path=None):
    if key_path is None:
        key_path = []
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = key_path + [key]
            if isinstance(value, str):
                cleaned_text = clean_response_string(value)
                st.write(f"Extracted from path {current_path}: {cleaned_text[:100]}...")
                return cleaned_text
            elif isinstance(value,(dict,list)): #Also check for nested lists
                result = extract_and_clean_text_from_dict(value, current_path)
                if result:
                    return result
        st.write(f"No string value found in dictionary at path: {key_path}")
        return "No text found within the dictionary"
    return None

def extract_and_clean_text_from_list(data):
    if isinstance(data, list):
        for item in data:
            if isinstance(item,str):
                cleaned_text = clean_response_string(item)
                st.write(f"Extracted from list: {cleaned_text[:100]}...")
                return cleaned_text
            elif isinstance(item,dict):
                result = extract_and_clean_text_from_dict(item)
                if result:
                    return result
            elif isinstance(item,list): #Check for nested lists
                result = extract_and_clean_text_from_list(item)
                if result:
                    return result

        st.write("No string or dict value found within the list")
        return "No text found within the list"
    return None

def generate_response(input_text):
    try:
        bot = get_chatbot()
        response = bot.ask(input_text)

        # Robust response handling
        if isinstance(response, str):
            response = clean_response_string(response)
        elif isinstance(response, dict):
            response = extract_and_clean_text_from_dict(response)
        elif isinstance(response, list):
            response = extract_and_clean_text_from_list(response)
        else:
            st.write(f"Unexpected response type: {type(response)}")
            response = "Sorry, the response format is not supported."

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, there was an error processing your request."

    return response

# Streamlit setup
st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

# Cache the Chatbot instance
@st.cache_resource
def get_chatbot(pdf_path='gpmc.pdf'):
    return CustomChatbot(pdf_path=pdf_path)


# Manage session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me questions about the document."}]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input
if input_text := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(input_text)

            if isinstance(response, str) and len(response) > 100:
                st.markdown(response)
            else:
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
