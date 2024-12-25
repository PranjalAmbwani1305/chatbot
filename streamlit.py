import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import time

load_dotenv()

HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")

if not HUGGINGFACE_API_KEY:
    st.error("Please set the HUGGINGFACE_API_KEY in your Streamlit secrets.")
    st.stop()
if not PINECONE_API_KEY:
    st.error("Please set the PINECONE_API_KEY in your Streamlit secrets.")
    st.stop()

os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

index_name = "chatbot"
pc = PineconeClient(api_key=PINECONE_API_KEY)

try:
    pc.create_index(
        name=index_name, dimension=768, metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Index '{index_name}' created successfully.")
except pinecone.core.client.exceptions.ApiException as e:
    if e.status == 409 and "Resource already exists" in str(e):
        try:
            existing_index = pc.describe_index(index_name)
            if existing_index.dimension == 768 and existing_index.metric == 'cosine':
                print(f"Index '{index_name}' already exists with the desired configuration. Skipping creation.")
            else:
                print(f"Index '{index_name}' exists but has a different configuration (dimension: {existing_index.dimension}, metric: {existing_index.metric}). Please delete it or use a different index name.")
                st.stop()
        except Exception as describe_err:
            print(f"Error describing index: {describe_err}")
            st.stop()
    else:
        print(f"Pinecone API Error: {e}")
        st.stop()

class CustomChatbot:
    def __init__(self, pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Reduced chunk size and added overlap
            self.docs = text_splitter.split_documents(documents)
            self.embeddings = HuggingFaceEmbeddings()
            self.index_name = index_name
            self.pc = pc
            start_time = time.time()
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            end_time = time.time()
            print(f"Pinecone Indexing Time: {end_time - start_time:.2f} seconds")
            self.retriever = self.docsearch.as_retriever(search_kwargs={"k": 3}) # Limit to top 3 results
            self.llm = HuggingFaceEndpoint(
                repo_id="distilbert-base-uncased-distilled-squad", #Consider smaller model if possible
                temperature=0.8, top_k=50,
                huggingfacehub_api_token=HUGGINGFACE_API_KEY
            )
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise

    def ask(self, question):
        start_time = time.time()
        try:
            context = self.retriever.get_relevant_documents(question)
            if not context:
                return "No relevant context found for this question."
            context_str = "\n".join([doc.page_content for doc in context])
            inputs = {
                "question": question,
                "context": context_str
            }
            response = self.llm.invoke(inputs)
            end_time = time.time()
            print(f"LLM inference time: {end_time - start_time:.2f} seconds")
            return response
        except Exception as e:
            st.error(f"Error during ask: {e}")
            return f"Error processing your request: {e}"

@st.cache_resource
def get_chatbot(pdf_path='gpmc.pdf'):
    try:
        return CustomChatbot(pdf_path=pdf_path)
    except Exception as e:
        st.error(f"Error creating chatbot: {e}")
        return None

def generate_response(input_text):
    chatbot = get_chatbot()
    if chatbot:
        return chatbot.ask(input_text)
    else:
        return "Failed to initialize the chatbot. Please check the logs."

st.set_page_config(page_title="Chatbot")
st.sidebar.title("Chatbot")

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
            response = generate_response(input_text)
            if isinstance(response, str) and len(response) > 100:
                st.markdown(response)
            else:
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
