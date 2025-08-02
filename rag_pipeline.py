from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from a .env file

load_dotenv()


def build_qa_chain(
    pdf_path: str,
    *,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o",
    k: int = 3,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> object:
    """
    Construct a retrieval‑augmented question answering chain for a given PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to index.
    embedding_model : str, optional
        Name of the OpenAI embedding model to use.  Defaults to
        ``text‑embedding‑3‑small`` which is the most recent cost‑efficient model
        as of July 2025【447204287098053†L0-L8】.
    chat_model : str, optional
        Name of the OpenAI chat model to use.  Defaults to ``gpt‑4o``, the
        current flagship model【719406984089018†L230-L243】.
    k : int, optional
        Number of documents to retrieve from the vector store.  Defaults to 3.
    chunk_size : int, optional
        Maximum length of each text chunk when splitting the PDF.  Defaults to
        1000 characters.
    chunk_overlap : int, optional
        Number of overlapping characters between consecutive chunks.  Defaults
        to 200 characters.

    Returns
    -------
    object
        A ``Runnable`` that can be invoked or streamed with a dictionary
        containing ``"input"`` and ``"chat_history"`` keys.  The result will
        include an ``"answer"`` key containing the generated response.
    """
    # Load and parse the PDF into LangChain Document objects
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the documents into manageable chunks. 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    # Create the embedding function
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Build an in‑memory Chroma vector store from the text chunks.  If you wish
    # to persist the vector store on disk, pass ``persist_directory`` here.
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Create a retriever that returns the top‐k most similar documents for a query.  
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Prompt to rewrite the user's question in a standalone form given the chat
    # history.  The context will be injected later by the chain.
    contextualize_q_prompt = ChatPromptTemplate.from_template(
        "Given the conversation so far and the latest user question, reformulate the question "
        "to be self-contained if needed.\n\nChat History:\n{chat_history}\n\nUser Question:\n{input}"
    )

    # Build the history‑aware retriever.  We set ``streaming=True`` to enable
    # token‑level streaming when using the chain's ``stream`` method.
    history_aware_retriever = create_history_aware_retriever(
        ChatOpenAI(model=chat_model, temperature=0, streaming=True),
        retriever,
        contextualize_q_prompt,
    )

    # Prompt to answer the user's question based on retrieved context.  The
    # ``context`` variable will be populated by ``create_retrieval_chain``.
    qa_prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {input}"
    )

    # Combine the chat model and prompt into a documents chain.  This chain
    # accepts a ``{"context": List[Document], "input": str}`` and returns a
    # string answer.
    question_answer_chain = create_stuff_documents_chain(
        ChatOpenAI(model=chat_model, temperature=0, streaming=True),
        qa_prompt,
    )

    # Assemble the retrieval chain.  The resulting chain will inject
    # ``chat_history`` into the history‑aware retriever and pass the retrieved
    # documents into the question answering chain.  The output is a dictionary
    # with at least ``"context"`` and ``"answer"`` keys【719406984089018†L230-L243】.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
