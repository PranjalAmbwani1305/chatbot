import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import time
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader, TextLoader # Import loaders

load_dotenv()

HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")

if not HUGGINGFACE_API_KEY:
    st.error("Please set the HUGGINGFACE_API_KEY in your Streamlit secrets.")
    st.stop()

os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY

index_name = "chatbot"
pc = None
if PINECONE_API_KEY:
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
    def __init__(self, documents):
        try:
            self.embeddings = HuggingFaceEmbeddings()
            self.index_name = index_name
            self.pc = pc

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            self.docs = text_splitter.split_documents(documents)

            if self.pc:
                start_time = time.time()
                self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
                end_time = time.time()
                print(f"Pinecone Indexing Time: {end_time - start_time:.2f} seconds")
                self.retriever = self.docsearch.as_retriever(search_kwargs={"k": 3})
            else: # Chroma as fallback if no Pinecone
                self.docsearch = Chroma.from_documents(self.docs, self.embeddings)
                self.retriever = self.docsearch.as_retriever(search_kwargs={"k": 3})

            self.llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                temperature=0.7, max_new_tokens=512,
                huggingfacehub_api_token=HUGGINGFACE_API_KEY
            )
            prompt_template = """Use the following context to answer the user's question. If you don't know the answer, just say "I don't know".

            Context:
            {context}

            Question: {question}

            Answer:
            """
            self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=self.retriever, chain_type_kwargs={"prompt": self.prompt}
            )

        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise

    def ask(self, question):
        try:
            start_time = time.time()
            result = self.qa_chain({"query": question})
            end_time = time.time()
            print(f"LLM inference time: {end_time - start_time:.2f} seconds")
            return result["result"].strip()
        except Exception as e:
            st.error(f"Error during ask: {e}")
            return f"Error processing your request: {e}"

@st.cache_resource
def get_chatbot(uploaded_file, use_pinecone):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyMuPDFLoader(uploaded_file)
            else: # Assumes text if not PDF
                loader = TextLoader(uploaded_file)
            documents = loader.load()
            return CustomChatbot(documents)
        else:
            return None
    except Exception as e:
        st.error(f"Error creating chatbot: {e}")
        return None

st.title("Document Chatbot")

uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
use_pinecone = st.checkbox("Use Pinecone (requires API key)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a document to start chatting!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if uploaded_file:
    chatbot = get_chatbot(uploaded_file, use_pinecone)
    if prompt := st.chat_input("Ask a question about the document:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant" and chatbot:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.ask(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
elif not uploaded_file and "messages" in st.session_state and len(st.session_state.messages) == 1:
        st.info("Please upload a document to begin.")
