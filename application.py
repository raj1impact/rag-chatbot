import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("📄 RAG PDF Chatbot")

query = st.text_input("Ask a question about the PDF")

if query:
    # Load and chunk PDF
    loader = PyPDFLoader("stouffville_summer-camps-guide_2024.pdf")
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(loader.load())

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_store_pdf")

    # LLM (Bedrock Claude via LangChain if configured, or local Ollama model)
    llm = OllamaLLM(model="mistral")  # Switch to Claude if using Bedrock
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Query
    response = qa_chain.invoke(query)
    st.success(response["result"])
