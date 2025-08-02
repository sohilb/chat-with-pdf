# 📄 Chat with PDF

**Chat with PDF** is a lightweight Streamlit app that lets you hold natural‑language conversations with the contents of any PDF document.  
Behind the scenes it spins up a Retrieval‑Augmented Generation (RAG) pipeline powered by [LangChain](https://github.com/langchain-ai/langchain), [OpenAI](https://openai.com/), and an in‑memory [Chroma](https://docs.trychroma.com/) vector store.

![Screenshot](./docs/screenshot.png)

---

## ✨ Features

| Capability | Description |
|------------|-------------|
| 🚀 **Zero‑setup PDF chat** | Drag‑and‑drop a PDF, ask questions, get answers. |
| 🧠 **Context‑aware retrieval** | Uses `create_history_aware_retriever` so follow‑up questions automatically carry conversation context. |
| 📚 **Chunking & embeddings** | Splits text with `RecursiveCharacterTextSplitter` and embeds with `OpenAIEmbeddings` (or your own). |
| ⚡ **Fast local vector store** | Chroma keeps everything in‑memory; no external DB required. |
| 🔐 **Env‑based secrets** | API keys read from `.env` – never hard‑coded. |
| 🛠 **Portable** | Works everywhere Python runs – deploy on Streamlit Cloud, Docker, or your laptop. |

---

## 🏁 Quickstart

```bash
# 1. Clone
git clone https://github.com/your‑org/chat‑with‑pdf.git
cd chat‑with‑pdf

# 2. Create & activate virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI key
echo "OPENAI_API_KEY=sk‑..." > .env    # or edit manually

# 5. Run the app
streamlit run app.py
```

Then open <http://localhost:8501> in your browser, upload a PDF, and start chatting! ✨

---

## 🗂️ Project Structure

```text
.
├── app.py            # Streamlit UI
├── rag_pipeline.py   # RAG pipeline builder
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All environment variables are loaded via **python‑dotenv** (`.env` file):

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key (used for embeddings & chat model) | ✅ |
| `OPENAI_MODEL` | Chat model name, defaults to `gpt-4o-mini` | optional |
| `EMBEDDING_MODEL` | Embedding model name, defaults to `text-embedding-3-large` | optional |

---

## 📝 Extending

- **Swap Embeddings** – Replace `OpenAIEmbeddings` with `HuggingFaceEmbeddings` for an open‑source stack.
- **Persistent Vector Store** – Point Chroma to a directory (e.g., `persist_directory="db"`) to keep indexes between restarts.
- **Multi‑PDF Support** – Hold multiple vectors in memory and add a selector drop‑down in `app.py`.
- **Download chat history** – Streamlit’s `st.download_button` makes it one‑liner.

See the inline comments in `rag_pipeline.py` for further pointers.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: langchain_community...` | Make sure you installed with `-r requirements.txt`. |
| `RateLimitError` | You may be sending too many requests to OpenAI; lower the `streaming=False`, `temperature=0`, or upgrade your plan. |
| “Please upload a PDF first” | Ensure `st.session_state.chain` is set (upload succeeded) – see your terminal for any indexing errors. |

---

## 📨 License

[MIT](LICENSE)

Made with ❤️ and lots of coffee.
