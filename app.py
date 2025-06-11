import os
import re
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Streamlit UI ---
st.title("Chat With Your PDF")

# Sidebar
# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="Enter your Groq API key here"
)

if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API key to use the model.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Select a model",
    ["deepseek-r1-distill-llama-70b", "qwen-qwq-32b", "llama3-70b-8192"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)

# Init Groq LLM
model = ChatGroq(model_name=model_choice, api_key=groq_api_key, temperature=temperature)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.expander("📄 PDF Viewer", expanded=False):
        pdf_viewer(file_bytes, annotations=[])

    # Save to temp file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Load PDF content
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents) # list of clean, overlapping chunks

    # Embedding + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    st.success("✅ PDF processed and ready for querying.")
    st.info("⏳ First-time load may take 20–40 seconds while the embedding model is being prepared.")

    # System Prompt
    system_prompt = (
        "You are an expert assistant tasked with answering questions using only the provided PDF content. "
        "Do not speculate or answer beyond the document. Cite specific facts when possible."
        "\n\nContext:\n{context}\n\n"
    )

    history_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    history_retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", history_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=history_retriever_prompt
    )

    qa_chain = create_stuff_documents_chain(
        llm=model,
        prompt=qa_prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=history_aware,
        combine_docs_chain=qa_chain
    )

    for msg in st.session_state.chat_history:
        role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg.content}")

    # User Query
    user_query = st.text_input("Ask the PDF:")
    if user_query:

        response = rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response['answer']))
        
        clean_answer = re.sub(r'<think>.*?</think>', '', response['answer'])

        st.subheader("Answer")
        st.write(clean_answer)
