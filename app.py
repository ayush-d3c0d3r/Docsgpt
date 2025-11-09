import os
import tempfile
import streamlit as st
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatCohere

def safe_get_generation_info(self, response):
    return {
        "response_id": getattr(response, "id", None),
        "finish_reason": getattr(response, "finish_reason", None),
        "token_count": getattr(response, "token_count", None), 
    }

ChatCohere._get_generation_info = safe_get_generation_info

os.environ["COHERE_API_KEY"] = "QaXkkMYgdg4PBwDBEHi0mt4kydBz2GbakWhNRrMp" 


llm = ChatCohere(model="command-r-plus", temperature=0.7, max_tokens=512)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

st.set_page_config(page_title="Ask from the Documents", layout="wide")
st.title(" Docs GPT")

tabs = st.tabs([" Ask the Docs", " History"])

if "history" not in st.session_state:
    st.session_state.history = []

def load_file(file) -> List[Document]:
    suffix = file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.read())
        tmp.flush()
        temp_path = tmp.name

    if suffix == "pdf":
        loader = PyPDFLoader(temp_path)
    elif suffix == "txt":
        loader = TextLoader(temp_path, encoding='utf8')
    else:
        return []

    docs = loader.load()
    os.remove(temp_path)
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunked

with tabs[0]:
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    question = st.text_input("Ask anything about the documents")

    if uploaded_files and question:
        with st.spinner("Processing..."):
            all_docs = []
            for file in uploaded_files:
                all_docs.extend(load_file(file))

            chunks = chunk_documents(all_docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )

            answer = qa_chain.run(question)

        st.markdown("###  Answer")
        st.write(answer)

        st.session_state.history.append((question, answer))

with tabs[1]:
    st.markdown("###  Question History")
    if not st.session_state.history:
        st.info("No questions asked yet.")
    else:
        for idx, (q, a) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{idx}:** {q}")
            st.markdown(f"**A{idx}:** {a}")
            st.markdown("---")

