import streamlit as st
from pdf_chatbot import extract_text_from_pdf, store_in_pinecone, query_pinecone

st.title("PDF Chatbot")

pdf_file = st.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    doc_id = str(hash(text))
    store_in_pinecone(text, doc_id)
    st.success("PDF text has been uploaded and stored in Pinecone.")

user_query = st.text_input("Ask a question about the PDF:")

if user_query:
    response = query_pinecone(user_query)
    st.write(f"Response: {response}")
