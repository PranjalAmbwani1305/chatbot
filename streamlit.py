import os
import re
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import login

# Hugging Face login (from Streamlit secrets)
login(token=st.secrets["HF_TOKEN"])

# Load environment variables
load_dotenv()

# Set up API keys from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class CustomChatbot:
    def __init__(self, pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
            self.docs = text_splitter.split_documents(documents)
            self.embeddings = HuggingFaceEmbeddings()
            self.index_name = "chatbot"
            self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))

            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name, dimension=768, metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )

            self.llm = HuggingFaceEndpoint(
                repo_id="distilbert-base-uncased-distilled-squad",
                temperature=0.8, top_k=50,
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )

            template = """
            You are a chatbot for answering questions about the specified document.
            Answer these questions and explain the process step by step.
            If you don't know the answer, just say "I don't know."

            Context: {context}
            Question: {question}
            Answer:
            """
            self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            self.retriever = self.docsearch.as_retriever() #Store the retriever
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise

    def ask(self, question):
        try:
            context = self.retriever.get_relevant_documents(question)
            if not context:
                return "No relevant context found for this question."

            context_str = "\n".join([doc.page_content for doc in context])
            formatted_prompt = self.prompt.format(context=context_str, question=question)
            st.write(f"Formatted Prompt:\n{formatted_prompt}")
            return self.llm.invoke(formatted_prompt)
        except Exception as e:
            st.error(f"Error during ask: {e}")
            return f"Error processing your request: {e}" #Include the exception message

def clean_response_string(text):
    if isinstance(text, str):
        return text.replace("\uf8e7", "").replace("\xad", "").replace("\\n", "\n").replace("\t", " ")
    return ""

def extract_text(data, path=""):
    # ... (extract_text function remains the same)

def generate_response(input_text):
    try:
        bot = get_chatbot()
        response = bot.ask(input_text)

        st.write(f"Raw Response Type: {type(response)}")
        st.write(f"Raw Response: {response}")

        extracted_response = extract_text(response)
        if not extracted_response.strip():
            return "Could not extract any text from the model's response."
        
        response = extracted_response

        # ... (formatting logic remains the same)

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return f"Sorry, there was an error processing your request: {e}" #Include the exception message

# Streamlit setup
# ... (Streamlit app code remains the same)
