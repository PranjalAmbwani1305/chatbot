import os
import re
import pinecone
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import PineconeClient, ServerlessSpec
from dotenv import load_dotenv
from huggingface_hub import login

# Streamlit secrets and environment variables
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT")

if not all([HUGGINGFACE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    st.error("Please set all API keys and environment in Streamlit secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Hugging Face Login (only if needed for private models)
try:
    login(token=HUGGINGFACE_API_KEY)  # Try logging in, but don't crash if it fails
except Exception as e:
    st.warning(f"Hugging Face login failed (this might be okay): {e}")

class Chatbot:
    def __init__(self, pdf_path="gpmc.pdf"): # Added default pdf_path
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Smaller chunks
            self.docs = text_splitter.split_documents(documents)
            self.embeddings = HuggingFaceEmbeddings()
            self.index_name = "amcgpmc"
            self.pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

            if self.index_name not in self.pc.list_indexes():
                self.pc.create_index(
                    name=self.index_name, dimension=768, metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1') #Check your region
                )

            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            self.retriever = self.docsearch.as_retriever(search_kwargs={"k": 3}) # Limit retrieved docs

            self.llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.7, # Adjusted temperature
                top_k=50,
            )

            template = """
            You are a helpful chatbot for the Ahmedabad Government Corporation (AMC). 
            Answer questions about the GPMC act in a clear and concise manner, providing step-by-step instructions where appropriate.
            If you don't know the answer based on the provided context, say "I'm sorry, I couldn't find information about that in the provided document."

            Context: {context}
            Question: {question}
            Answer:
            """
            self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

            self.rag_chain = (
                {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise # Important to re-raise so Streamlit catches the error

    def ask(self, question):
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            st.error(f"Error during query: {e}")
            return "An error occurred while processing your request."

# Streamlit app
st.set_page_config(page_title="GPMC Chatbot")
st.title("GPMC Act Chatbot")

@st.cache_resource
def get_chatbot(pdf_path):
    try:
        return Chatbot(pdf_path)
    except Exception as e:
        st.error(f"Error creating chatbot: {e}")
        return None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me questions about the GPMC Act."}]

pdf_path = st.file_uploader("Upload PDF", type="pdf")

if pdf_path:
    chatbot = get_chatbot(pdf_path)
else:
    st.info("Please upload a PDF document to begin.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            if chatbot: #Check if the chatbot was initialized successfully
                response = chatbot.ask(user_input)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Chatbot initialization failed. Please try again.")
