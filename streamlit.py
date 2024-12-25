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
login(token='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')

# Load environment variables
load_dotenv()

# Set up API keys
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class CustomChatbot:
    def __init__(self, pdf_path):
        try:
            # Load PDF documents
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            # Split documents into smaller chunks
            text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
            self.docs = text_splitter.split_documents(documents)

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings()

            # Pinecone setup
            self.index_name = "chatbot"
            self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))

            # Create Pinecone index if it doesn't exist
            if self.index_name not in [index.name for index in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=736,  # Adjust dimension based on your embeddings
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )

            # Load Pinecone index
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

            # Setup HuggingFace model for Q&A
            self.llm = HuggingFaceEndpoint(
                repo_id="distilbert-base-uncased-distilled-squad",
                temperature=0.8,
                top_k=50,
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )

            # Define the prompt template
            template = """
            You are a chatbot for answering questions about the specified document. 
            Answer these questions and explain the process step by step.
            If you don't know the answer, just say "I don't know."

            Context: {context}
            Question: {question}
            Answer: 
            """
            self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

            # Define the retrieval-augmented generation (RAG) chain
            self.rag_chain = (
                {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise

    def ask(self, question):
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process your question."

# Streamlit setup
st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

# Cache the Chatbot instance to avoid reloading the model and data each time
@st.cache_resource
def get_chatbot():
    return CustomChatbot(pdf_path='gpmc.pdf')

# Function to generate response from the chatbot
def generate_response(input_text):
    try:
        bot = get_chatbot()
        response = bot.ask(input_text)

        # Clean response from any unwanted characters
        if isinstance(response, str):
            response = response.replace("\uf8e7", "").replace("\xad", "").replace("\\n", "\n").replace("\t", " ")
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, there was an error processing your request."

# Manage session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions about the document."}
    ]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input and generate response
if input_text := st.chat_input("Type your question here..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(input_text)

            # Display the response
            if isinstance(response, str) and len(response) > 100:
                st.markdown(response)
            else:
                st.write(response)

        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
