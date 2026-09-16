import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = "data/docs"
VECTORSTORE_DIR = "vectorstore/chroma_db"

def build_vectorstore():
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    print("1/3 Loading documentation and technical manuals...")
    text_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = text_loader.load() + pdf_loader.load()
    if not docs:
        print(f"No documents found in {DOCS_DIR}. Please add technical manuals/logs.")
        return None

    print(f"2/3 Chunking {len(docs)} document(s)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    print("3/3 Generating embeddings and persisting to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=VECTORSTORE_DIR
    )
    print(f"SUCCESS! Vectorstore persisted at: {os.path.abspath(VECTORSTORE_DIR)}")
    return vectorstore

if __name__ == "__main__":
    build_vectorstore()