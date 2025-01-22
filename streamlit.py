import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
from huggingface_hub import login
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Authenticate Hugging Face
login(token='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')

# Load environment variables
load_dotenv()

# Environment variable validation
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not HUGGINGFACE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys for Hugging Face or Pinecone are missing.")

# Set API keys
os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY


class Chatbot:
    def __init__(self):
        logging.debug("Initializing Chatbot...")

        try:
            # Load and split documents
            loader = PyMuPDFLoader('gpmc.pdf')
            documents = loader.load()
            logging.debug(f"Loaded documents: {documents[:2]}")  # Log the first 2 docs for preview

            text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
            self.docs = text_splitter.split_documents(documents)
            logging.debug(f"Split documents into chunks: {len(self.docs)} chunks")

            # Initialize embeddings and vector store
            self.embeddings = HuggingFaceEmbeddings()
            self.index_name = "chatbot"
            self.pc = PineconeClient(api_key=PINECONE_API_KEY)

            # Create Pinecone index if not exists
            if self.index_name not in self.pc.list_indexes().names():
                logging.debug(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )

            # Populate the vector store
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

            # Initialize Hugging Face LLM
            repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            self.llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                temperature=0.8,
                top_k=50,
                huggingfacehub_api_token=HUGGINGFACE_API_KEY
            )
            logging.debug("HuggingFace LLM initialized.")

            # Define prompt template
            template = """
            Given the context below, answer the question. Be as precise as possible and provide detailed information from the context if available.

            Context: {context}

            Question: {question}

            Answer:
            """
            self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

            # Define RAG chain
            self.rag_chain = (
                {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

        except Exception as e:
            logging.error(f"Error during chatbot initialization: {e}")
            raise

    def ask(self, question):
        logging.debug(f"Asking question: {question}")
        try:
            response = self.rag_chain.invoke(question)
            logging.debug(f"Response: {response}")
            return response
        except Exception as e:
            logging.error(f"Error during chatbot response generation: {e}")
            raise


@st.cache_resource
def get_chatbot():
    """Cache chatbot instance for reuse."""
    return Chatbot()


def format_response(response):
    """Format the chatbot's response for better readability."""
    if isinstance(response, str):
        # Clean and standardize the response
        response = response.replace("\uf8e7", "").replace("\xad", "")
        response = response.replace("\\n", "\n").replace("\n", " ")
        response = response.replace("Guj", "Gujarat")

        # Split text into sentences or list-like structure
        lines = response.split(". ")
        formatted_response = []

        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            # Detect numbered or bulleted lists
            if line[0].isdigit() or line.startswith(("-", "*")):
                formatted_response.append(f"- {line}")
            else:
                formatted_response.append(line)

        # Join the lines into a readable format
        return "\n\n".join(formatted_response)

    return response


# Initialize Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Ask me questions about the GPMC of AMC."}
    ]

# Display conversation history
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
            try:
                chatbot = get_chatbot()
                response = chatbot.ask(input_text)
                formatted_response = format_response(response)

                if len(formatted_response) > 100:
                    st.markdown(formatted_response)
                else:
                    st.write(formatted_response)

                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            except Exception as e:
                logging.error(f"Error during chatbot response generation: {e}")
                st.error("Sorry, there was an error generating the response.")

