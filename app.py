from __future__ import annotations

import os
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import streamlit as st

from rag_pipeline import build_qa_chain

# Configure the Streamlit page
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.sidebar.title("Upload a PDF")

# Initialize session state variables.  ``chat_history`` holds tuples of
# (role, message) where role is "human" or "ai".  ``chain`` stores the
# retrieval‑augmented chain for the current PDF.  ``pdf_name`` tracks the
# name of the last uploaded file to detect when a new upload occurs.
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str]] = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Handle PDF uploads via the sidebar.  When a new PDF is uploaded the
# existing chat history is cleared and a new chain is built.  Any errors
# encountered during indexing are surfaced to the user.
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
    # Reset the chat history when switching documents
    st.session_state.chat_history = []
    # Write the uploaded PDF to a temporary file on disk
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
    try:
        # Build a new QA chain for the uploaded PDF
        st.session_state.chain = build_qa_chain(pdf_path)
        st.session_state.pdf_name = uploaded_file.name
        st.sidebar.success(f"PDF '{uploaded_file.name}' indexed successfully!")
    except Exception as exc:
        # Surface any errors encountered during indexing
        st.session_state.chain = None
        st.sidebar.error(f"Failed to index PDF: {exc}")
    finally:
        # Clean up the temporary file
        try:
            os.remove(pdf_path)
        except OSError:
            pass

# Chat interface header
st.header("Ask questions about your PDF")

# Display the existing chat history.  Map our internal roles ("human"/"ai")
# to Streamlit roles ("user"/"assistant") for rendering.
for role, message in st.session_state.chat_history:
    streamlit_role = "user" if role == "human" else "assistant"
    with st.chat_message(streamlit_role):
        st.markdown(message)

# Render the chat input.  It is disabled and displays a helpful placeholder
# until a PDF has been indexed.  Once a chain exists the placeholder
# prompts the user to ask a question about the PDF.
prompt = st.chat_input(
    "Ask a question about the PDF" if st.session_state.chain is not None else "Please upload a PDF first.",
    disabled=st.session_state.chain is None,
)

# Process the user’s question only if there is input and a chain exists
if prompt and st.session_state.chain is not None:
    # Append the user's question to chat history as a human message
    st.session_state.chat_history.append(("human", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    chain = st.session_state.chain

    # Convert chat history to the format expected by the chain
    lc_history = []
    for role, msg in st.session_state.chat_history:
        # Use the roles "human" and "ai" expected by LangChain
        if role in {"human", "ai"}:
            lc_history.append((role, msg))
        else:
            # Fall back to mapping Streamlit roles to LangChain roles
            lc_history.append(("human" if role in {"user"} else "ai", msg))

    # Stream the assistant's answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        for chunk in chain.stream({
            "input": prompt,
            "chat_history": lc_history,
        }):
            # Each chunk is a dictionary with at least an "answer" key【795567596223965†L70-L93】
            if "answer" in chunk:
                full_answer += chunk["answer"]
                # Update the placeholder with the incremental answer
                placeholder.markdown(full_answer)
        # Append the assistant's full answer to the chat history
        st.session_state.chat_history.append(("ai", full_answer))