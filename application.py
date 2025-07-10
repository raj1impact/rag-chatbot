import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockLLM  # ✅ Use Bedrock

st.title("📄 RAG PDF Chatbot (Claude via Bedrock)")

query = st.text_input("Ask a question about the PDF")

if query:
    # Load and chunk PDF
    loader = PyPDFLoader("stouffville_summer-camps-guide_2024.pdf")
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(loader.load())

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_store_pdf")

    # LLM using Claude from Bedrock
    llm = BedrockLLM(model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1")

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Query
    response = qa_chain.invoke(query)
    st.success(response["result"])
