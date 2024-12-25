import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = st.secrets.get("HUGGINGFACE_API_KEY")
os.environ['PINECONE_API_KEY'] = st.secrets.get("PINECONE_API_KEY")

class CustomChatbot:
    def __init__(self, pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
            self.docs = text_splitter.split_documents(documents)
            self.embeddings = HuggingFaceEmbeddings()
            self.index_name = "chatbot"  # Or use a more dynamic name if needed
            self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))

            try:
                self.pc.create_index(
                    name=self.index_name, dimension=768, metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                print(f"Index '{self.index_name}' created successfully.")
            except pinecone.core.client.exceptions.ApiException as e:
                if e.status == 409 and "Resource already exists" in e.body:
                    try:
                        existing_index = self.pc.describe_index(self.index_name)
                        if existing_index.dimension == 768 and existing_index.metric == 'cosine':
                            print(f"Index '{self.index_name}' already exists with the desired configuration. Skipping creation.")
                        else:
                            print(f"Index '{self.index_name}' exists but has a different configuration. Please delete it or use a different index name.")
                            raise  # Re-raise to prevent initialization
                    except Exception as describe_err:
                        print(f"Error describing index: {describe_err}")
                        raise  # Re-raise to prevent initialization
                else:
                    raise  # Re-raise other exceptions

            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            self.retriever = self.docsearch.as_retriever()
            self.llm = HuggingFaceEndpoint(
                repo_id="distilbert-base-uncased-distilled-squad",
                temperature=0.8, top_k=50,
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            raise  # Important: re-raise the exception to stop execution

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
            response = self.llm.invoke(inputs)
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
st.title("Chatbot")

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
