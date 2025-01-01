import os
import time
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

login(token='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')

load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class Chatbot:
    def __init__(self):
        loader = PyMuPDFLoader('gpmc.pdf') 
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        self.docs = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings()

        self.index_name = "chatbot"
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY')) 
        
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, 
            temperature=0.8, 
            top_k=50, 
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        template = """
        Given the context below, answer the question. Be as precise as possible and provide detailed information from the context if available.

        Context: {context}

        Question: {question}

        Answer:
        """
        self.prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)
    

@st.cache_resource
def get_chatbot():
    return Chatbot()

def generate_response(input_text):
    bot = get_chatbot()
    response = bot.ask(input_text)

    if isinstance(response, str):
        response = response.replace("\uf8e7", "").replace("\xad", "")
        response = response.replace("\\n", "\n").replace("\t", " ")
        response = response.replace("Guj", "Gujarat")

        response_parts = response.split("\n")
        formatted_response = []
        current_part = ""
        
        for part in response_parts:
            if part.strip().isdigit() or part.strip().startswith(tuple(str(i) for i in range(1, 10))) or part.strip().startswith(("404.", "405.")):
                if current_part:
                    formatted_response.append(current_part.strip())
                current_part = f"{part.strip()} "
            else:
                current_part += part.strip() + " "
        
        if current_part:
            formatted_response.append(current_part.strip())
        
        return "\n\n".join(f"- {part}" for part in formatted_response)
    
    return response

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Ask me questions about the GPMC of AMC."}
    ]

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
