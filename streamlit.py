#login(token = "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")
import os
import re
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API keys from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets.get("HUGGINGFACE_API_KEY") # Use .get to avoid KeyError
os.environ['PINECONE_API_KEY'] = st.secrets.get("PINECONE_API_KEY")

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
            # THIS IS THE KEY CHANGE: Connect to the Pinecone index as a vectorstore
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            self.retriever = self.docsearch.as_retriever()

            self.llm = HuggingFaceEndpoint(
                repo_id="distilbert-base-uncased-distilled-squad",
                temperature=0.8, top_k=50,
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )

        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise  # Re-raise the exception after logging it in Streamlit

    def ask(self, question):
        try:
            context = self.retriever.get_relevant_documents(question)
            if not context:
                return "No relevant context found for this question."

            context_str = "\n".join([doc.page_content for doc in context])

            inputs = {
                "question": question,
                "context": context_str
            }
            #st.write(f"Inputs:\n{inputs}")  # Debugging: Print the inputs
            response = self.llm.invoke(inputs)
            return response
        except Exception as e:
            st.error(f"Error during ask: {e}")
            return f"Error processing your request: {e}"

# ... (rest of the helper functions: clean_response_string, extract_text, generate_response)

# Streamlit setup
st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

@st.cache_resource
def get_chatbot(pdf_path='gpmc.pdf'):
    try:
        return CustomChatbot(pdf_path=pdf_path)
    except Exception as e:
        st.error(f"Error creating chatbot: {e}")
        return None  # Return None if chatbot creation fails

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me questions about the document."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if input_text := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            chatbot = get_chatbot() # Get the chatbot from the cache
            if chatbot: # Only proceed if chatbot creation was successful
                response = generate_response(input_text)
                if isinstance(response, str) and len(response) > 100:
                    st.markdown(response)
                else:
                    st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Failed to initialize the chatbot. Please check the logs.")
                st.stop() # Stop execution if the chatbot isn't created.
