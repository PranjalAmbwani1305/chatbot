import os
import time
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import Pinecone as PineconeClient, ServerlessSpec

login(token='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')

load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

@st.cache_resource
def load_and_process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
    return text_splitter.split_documents(documents)

class CustomChatbot:
    def __init__(self, pdf_path):
        self.docs = load_and_process_pdf(pdf_path)
        self.embeddings = HuggingFaceEmbeddings()
        self.index_name = "chatbot"
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))
        
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        self.llm = HuggingFaceEndpoint(
            repo_id="distilbert-base-uncased-distilled-squad",
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

        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}

            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        try:
            inputs = {"context": self.docsearch.as_retriever(), "question": question}
            return self.rag_chain.invoke(inputs)
        except Exception as e:
            st.error(f"Error during RAG chain execution: {e}")
            return "Sorry, an error occurred while generating the response."

def get_chatbot(pdf_path='gpmc.pdf'):
    return CustomChatbot(pdf_path=pdf_path)

def generate_response(input_text):
    try:
        start_time = time.time()

        bot = get_chatbot()  
        load_time = time.time()

        response = bot.ask(input_text)  
        ask_time = time.time()

        st.write(f"Time to load chatbot: {load_time - start_time:.2f}s")
        st.write(f"Time to generate response: {ask_time - load_time:.2f}s")

        if isinstance(response, str):
            response = response.replace("\uf8e7", "").replace("\xad", "").replace("\\n", "\n").replace("\t", " ")
        elif isinstance(response, dict):
            response_text = response.get('text', "No meaningful response found.")
            response = response_text.replace("\uf8e7", "").replace("\xad", "").replace("\\n", "\n").replace("\t", " ")
    except Exception as e:
        st.error(f"Error during response generation: {e}")
        response = "Sorry, there was an error processing your request."

    return response

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me questions about the document."}
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
