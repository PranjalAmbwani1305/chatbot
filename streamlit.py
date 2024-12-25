import os
from langchain_core.prompts import PromptTemplate
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

        loader = PyMuPDFLoader(pdf_path) 
        documents = loader.load()
        

        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)
        

        self.embeddings = HuggingFaceEmbeddings()
      
        self.index_name = "chatbot"
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY')) 
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=348,  
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'  
                )
            )

  
        self.llm = HuggingFaceEndpoint(
            repo_id="deepset/roberta-base-squad2", 
            temperature=0.8, 
            top_k=50, 
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
        self.prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # Initialize Pinecone index with documents
        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

        # Define the retrieval-augmented generation (RAG) chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)



st.set_page_config(page_title=" Chatbot")

    st.title("Chatbot")

# Cache the Chatbot instance
@st.cache_resource
def get_chatbot():
    return CustomChatbot(pdf_path='gpmc.pdf')


def generate_response(input_text):
    bot = get_chatbot()
    response = bot.ask(input_text)

 
    if isinstance(response, str):
        response = response.replace("\uf8e7", "").replace("\xad", "")
        response = response.replace("\\n", "\n").replace("\t", " ")  

        return response
    
    return response


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": " Ask me questions about the document."}
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if input_text := st.chat_input("Type your question here..."):
    # Append user message to session state
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
