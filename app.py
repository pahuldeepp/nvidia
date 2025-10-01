import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load .env file
load_dotenv()

# Initialize NVIDIA LLM
llm = ChatNVIDIA(
    api_key=os.getenv("api_key"),   # make sure .env has api_key=nvapi-xxxx
    model="meta/llama-4-maverick-17b-128e-instruct",
    base_url="https://integrate.api.nvidia.com/v1"
)

st.title("📘 NVIDIA NIM RAG Demo")
st.write("Upload a PDF, embed it with NVIDIA Embeddings, and ask questions!")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def vector_embedding(file_path):
    """Load PDF, split, embed, and store in FAISS."""
    if "vectors" not in st.session_state:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=70
        )
        final_docs = text_splitter.split_documents(docs)

        embeddings = NVIDIAEmbeddings(
            api_key=os.getenv("api_key"),
            model="nvidia/nv-embedqa-e5-v5",   # NVIDIA Embedding model
            base_url="https://integrate.api.nvidia.com/v1"
        )

        st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

# Prompt template
prompt = PromptTemplate.from_template("""
Answer the question based on the context provided. Please provide the most relevant answer.

<context>
{context}
</context>

Question: {input}
""")

# Input box
prompt1 = st.text_input("💬 Enter your question")

# Build vector DB if button clicked
if uploaded_file and st.button("📥 Process Document"):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    vector_embedding("temp.pdf")
    st.success("✅ FAISS vector store DB is ready using NVIDIA AI Endpoints")

# Run QA if user asks a question
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    st.write("⏱️ Time taken: ", time.process_time() - start)

    # Answer
    st.subheader("🔎 Answer")
    st.write(response.get("answer") or response.get("output_text"))

    # Show relevant context docs
    with st.expander("📑 Relevant Document Passages"):
        for doc in response.get("context") or response.get("source_documents", []):
            st.write(doc.page_content)
            st.write("---")
