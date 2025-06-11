📄 Chat With Your PDF
A Retrieval-Augmented Generation (RAG) web app that allows users to upload and query PDF files using advanced LLMs. Built with LangChain, Groq, FAISS, and Streamlit, this app provides accurate, context-aware answers from the uploaded document.

🚀 Features
🔍 PDF Upload & Parsing: Upload any PDF and extract its contents.

🧩 Smart Text Chunking: Uses RecursiveCharacterTextSplitter for robust document segmentation.

📚 Vector-Based Retrieval: FAISS vector store powered by sentence-transformers embeddings.

🧠 History-Aware Chat: Reformulates follow-up questions based on chat history.

🤖 LLM Integration via Groq: Supports LLaMA3, DeepSeek, and Qwen models via Groq API.

🧵 Streamlit Chat UI: Real-time conversational interface.

✨ Clean Responses: Strips hallucinated or irrelevant LLM behavior.

📸 Demo
<!-- Optional: Add your app screenshot here -->

🛠️ Tech Stack
Component	Library/Tool
Frontend	Streamlit
LLM Backend	LangChain + Groq LLMs
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Store	FAISS
PDF Loader	LangChain's PyPDFLoader

🧪 Setup Instructions
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/your-username/chat-with-your-pdf.git
cd chat-with-your-pdf
2. Create a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add your Groq API Key
Create a .env file in the root folder:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
5. Run the app
bash
Copy
Edit
streamlit run app.py
📁 File Structure
bash
Copy
Edit
.
├── app.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
├── /demo                    # Screenshots or demos (optional)
└── README.md                # You're here!
⚙️ Models Supported
deepseek-r1-distill-llama-70b

llama-guard-3-8b

qwen-qwq-32b

These models are queried via the Groq API.

🔐 Environment Variables
Variable	Description
GROQ_API_KEY	Your Groq LLM API Key

📌 Notes
No cloud storage: PDF is processed locally in memory using tempfile.

Embeddings and retrieval are done per session.

Only supports one PDF at a time per session.

