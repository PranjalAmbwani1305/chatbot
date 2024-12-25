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
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Ensure this is correct for your embeddings
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        # Setup HuggingFace model for Q&A (using a transformer model like RoBERTa)
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
        try:
            self.prompt = PromptTemplate(
                template=template, 
                input_variables=["context", "question"]
            )
        except Exception as e:
            st.error(f"Error initializing PromptTemplate: {e}")
            raise  # Re-raise the exception to stop further execution if necessary

        # Initialize Pinecone index with documents
        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

        # Define the retrieval-augmented generation (RAG) chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough() }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        # Prepare the inputs as a dictionary
        context = self.docsearch.as_retriever()
        return self.rag_chain.invoke({"context": context, "question": question})


# Streamlit setup
st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

# Cache the Chatbot instance to avoid reloading the model and data each time
def get_chatbot(pdf_path='gpmc.pdf'):
    # Initialize chatbot only once to avoid reloading large data
    return CustomChatbot(pdf_path=pdf_path)


# Function to generate response from the chatbot
def generate_response(input_text):
    try:
        bot = get_chatbot()  # Get or initialize the chatbot instance
        response = bot.ask(input_text)  # Get the response

        # Check if the response is a string before using replace()
        if isinstance(response, str):
            response = response.replace("\uf8e7", "").replace("\xad", "")
            response = response.replace("\\n", "\n").replace("\t", " ")  # Clean newlines and tabs
        elif isinstance(response, dict):
            # If response is a dictionary, extract the relevant value (e.g., 'text' key)
            response_text = response.get('text', "Sorry, no text found in response.")
            st.write(f"Extracted Text from Dict: {response_text}")
            response = response_text  # Set response to the extracted text
    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, there was an error processing your request."

    return response


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

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(input_text)  # Get the response from the chatbot

            # Display the formatted response
            if isinstance(response, str) and len(response) > 100:
                st.markdown(response)  # Display the response as markdown if it's a long response
            else:
                st.write(response)  # Display the response as plain text for shorter responses

        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
