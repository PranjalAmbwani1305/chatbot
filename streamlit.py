from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv
import os
import pinecone

# Load environment variables
load_dotenv()
os.environ['HUGGINGFACEHUB_API_KEY] = os.getenv("HUGGINGFACEHUB_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"  # Replace with your Pinecone environment

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "chatbot"

# Create or connect to Pinecone index
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # Adjust dimension as per your model
pinecone_index = pinecone.Index(index_name)

# HuggingFace Embeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# PyPDFLoader
Pdfloader = PyPDFLoader("gpmc.pdf")  # Replace with your PDF file path
Pdfdocuments = Pdfloader.load()
Pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = Pdf_text_splitter.split_documents(Pdfdocuments)

# Store PDF documents in Pinecone
pdf_vectordb = Pinecone.from_documents(final_documents[:120], hf_embeddings, index=pinecone_index)
PdfRetriever = pdf_vectordb.as_retriever()
PdfRetriever_tool = create_retriever_tool(
    PdfRetriever,
    name="PDF Retriever",
    description="Retrieve information from the uploaded PDF."
)

# Tools (only PDF retrieval)
tools = [PdfRetriever_tool]

# Language Model (HuggingFaceHub)
hf_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B",
    model_kwargs={"temperature": 0.1, "max_length": 300}
)

# Agent Setup
agent_executor = AgentExecutor(agent=create_tool_calling_agent(hf_llm, tools), tools=tools, verbose=True)

# User Input and Execution
user_input = input("Enter Search Query: ")
agent_executor.invoke({"input": user_input})
