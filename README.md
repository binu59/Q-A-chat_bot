# 🧠 DocMind — AI-Powered PDF Chat

DocMind is a Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and have a conversation with it. Built with LangChain, Google Gemini, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-orange?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

- 📄 Upload any PDF and instantly start asking questions
- 🤖 Powered by **Google Gemini 2.5 Flash** as the LLM
- 🔍 **Local embeddings** via `all-MiniLM-L6-v2` — no API limits, no rate errors
- 🧠 **Conversation memory** — ask follow-up questions naturally
- 📎 **Source citations** — every answer shows which pages it came from
- ⚡ Smart caching — re-uploading the same PDF doesn't rebuild the pipeline
- 🛡️ Retry logic — gracefully handles transient LLM errors

---

##  How It Works

```
PDF Upload
    ↓
Extract text (PyPDFLoader)
    ↓
Split into chunks (RecursiveCharacterTextSplitter)
    ↓
Embed chunks locally (all-MiniLM-L6-v2 via HuggingFace)
    ↓
Store in vector database (ChromaDB)
    ↓
User asks a question
    ↓
Embed question locally → similarity search → retrieve top 6 chunks
    ↓
Send chunks + question + chat history → Gemini 2.5 Flash
    ↓
Answer with source page references
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain + LangChain Classic |
| PDF Loader | PyPDFLoader |
| Memory | ConversationBufferMemory |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docmind.git
cd docmind
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `all-MiniLM-L6-v2` 

### 4. Set up your API key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit
python-dotenv
langchain
langchain-community
langchain-core
langchain-classic
langchain-chroma
langchain-google-genai
langchain-huggingface
sentence-transformers
chromadb
pypdf
```

---

## 📁 Project Structure

```
docmind/
├── app.py              # Main Streamlit application
├── .env                # API keys (never commit this)
├── .gitignore          # Excludes .env and cache files
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

---

## ⚙️ Configuration

You can tweak these values in `app.py` to adjust performance:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Characters per text chunk |
| `chunk_overlap` | `150` | Overlap between chunks |
| `k` (retriever) | `6` | Number of chunks retrieved per question |
| `max_retries` | `3` | Retry attempts on LLM errors |

---







## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.
