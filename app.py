from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

# --- Streamlit UI ---
st.title("Chat With Your PDF")

# Sidebar
model_choice = st.sidebar.selectbox(
    "Select a model",
    ["deepseek-r1-distill-llama-70b", "llama-guard-3-8b", "qwen-qwq-32b"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)

# Init Groq LLM
model = ChatGroq(model_name=model_choice, api_key=os.getenv("GROQ_API_KEY"), temperature=temperature)

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.subheader("📄 PDF Viewer")
    pdf_viewer(file_bytes, annotations=[])

    # Save to temp file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Load PDF content
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents) # list of clean, overlapping chunks

    # Embedding + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed and ready for querying.")

    # User Query
    user_query = st.text_input("Ask the PDF:")
    if user_query:
        relevant_docs = vectorstore.similarity_search(user_query, k=3)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        with st.expander("📚 Retrieved Context"):
            st.write(context)

        response = model.invoke([
            SystemMessage(content="You are a helpful assistant. Use the following context to answer the user's question."),
            HumanMessage(content=f"{user_query}\n\nContext:\n{context}")
        ])
        
        st.subheader("Answer")
        st.write(response.content)

