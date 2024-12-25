import os
import re
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

login(token = "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")

# Load environment variables
load_dotenv()

# Set up API keys
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

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

            self.llm = HuggingFaceEndpoint(
                repo_id="distilbert-base-uncased-distilled-squad",
                temperature=0.8, top_k=50,
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
            self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            self.rag_chain = (
                {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt | self.llm | StrOutputParser()
            )
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise  # Re-raise to stop execution

    def ask(self, question):
        try:
            context = self.docsearch.as_retriever()
            inputs = {"context": context, "question": question}
            return self.rag_chain.invoke(inputs)
        except Exception as e:
            st.error(f"Error during ask: {e}")
            return "Error processing your request."

def clean_response_string(text):
    if isinstance(text, str):
        return text.replace("\uf8e7", "").replace("\xad", "").replace("\\n", "\n").replace("\t", " ")
    return ""

def extract_text(data, path=""):
    if isinstance(data, str):
        return clean_response_string(data)
    elif isinstance(data, (dict, list)):
        extracted_text = ""
        if isinstance(data, dict):
            items = data.items()
        else:
            items = enumerate(data)

        for key, value in items:
            current_path = f"{path}[{key}]" if path else str(key)
            if isinstance(value, (str, dict, list)):
                result = extract_text(value,current_path)
                if result:
                    st.write(f"Extracted from {current_path}: {result[:100]}...")
                    extracted_text += result + "\n"
        return extracted_text
    return ""

def generate_response(input_text):
    try:
        bot = get_chatbot()
        response = bot.ask(input_text)

        st.write(f"Raw Response Type: {type(response)}")
        st.write(f"Raw Response: {response}")

        extracted_response = extract_text(response)
        if not extracted_response.strip():
            return "Could not extract any text from the model's response."
        
        response = extracted_response

        response_parts = response.split("\n")
        formatted_response = []
        current_part = ""

        for part in response_parts:
            if not part.strip():
                continue

            if re.match(r"^\d+\.", part.strip()) or re.match(r"^\d+$", part.strip()) or part.strip().startswith(("404.", "405.")):
                if current_part:
                    formatted_response.append(current_part.strip())
                current_part = f"{part.strip()} "
            else:
                current_part += part.strip() + " "

        if current_part:
            formatted_response.append(current_part.strip())

        return "\n\n".join(f"- {part}" for part in formatted_response)

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, there was an error processing your request."

# Streamlit setup
st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

@st.cache_resource
def get_chatbot(pdf_path='gpmc.pdf'):  # Replace 'gpmc.pdf' with your PDF file
    return CustomChatbot(pdf_path=pdf_path)

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
