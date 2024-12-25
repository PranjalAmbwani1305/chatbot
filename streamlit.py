import streamlit as st
import warnings
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import pinecone

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Pinecone index name
index_name = "my-vector-index"  # Change this to your Pinecone index name

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone vector store
vector_store = Pinecone(index_name=index_name, embedding_function=embedding_model.embed_query)

# Initialize HuggingFace LLM
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens": 1024},
)

# Simplified Prompt Template (generic QA prompt)
prompt_template = """
Context: {context}
Question: {question}
Answer:
"""

# Set up the retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(top_k=3),
    chain_type_kwargs={"prompt": prompt_template},
)

# Streamlit UI
st.title("AI-Powered Knowledge Assistant")

# File Upload Section
uploaded_file = st.file_uploader("Upload your PDF file for knowledge embedding", type="pdf")
if uploaded_file:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    with st.spinner("Processing your file..."):
        # Load and split PDF
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        texts = text_splitter.split_documents(documents)

        # Add documents to Pinecone
        vector_store.add_documents(documents=texts)
        st.success("File successfully processed and added to the vector store!")

# Query Section
query = st.text_input("Enter your query:")
if query:
    with st.spinner("Fetching the response..."):
        result = qa_chain.run(query)
        st.write("### Answer:")
        st.write(result)
