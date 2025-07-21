import os
import pickle
from pathlib import Path
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from modules.logger import logger

FAISS_INDEX_PATH = "./faiss_index"
UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Cache the embedder ===
_cached_embedding = None
def get_embedder():
    global _cached_embedding
    if _cached_embedding is None:
        _cached_embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={"device": "cpu"}
        )
        _ = _cached_embedding.embed_query("warmup")
        logger.info("✅ HuggingFace embedder cached and ready")
    return _cached_embedding

# === Main loader ===
def load_vectorstore(in_memory_files, overwrite=True):
    logger.info("📂 Starting file upload and vectorization...")

    # Step 1: Save files
    file_paths = []
    for filename, content in in_memory_files:
        if not content:
            continue
        save_path = Path(UPLOAD_DIR) / filename
        with open(save_path, "wb") as f:
            f.write(content)
        file_paths.append(save_path)

    logger.info(f"✅ Saved {len(file_paths)} file(s) to {UPLOAD_DIR}")

    # Step 2: Load documents
    docs = []
    for path in file_paths:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
            elif ext == ".txt":
                docs.extend(TextLoader(str(path), encoding="utf8").load())
            elif ext in (".doc", ".docx"):
                docs.extend(UnstructuredWordDocumentLoader(str(path)).load())
            elif ext in (".ppt", ".pptx"):
                docs.extend(UnstructuredPowerPointLoader(str(path)).load())
            elif ext in (".xls", ".xlsx"):
                df = pd.read_excel(path)
                csv_text = df.to_csv(index=False)
                docs.append(Document(page_content=csv_text, metadata={"source": path.name}))
            else:
                docs.extend(TextLoader(str(path), encoding="utf8").load())
        except Exception as e:
            logger.error(f"❌ Failed to load {path.name}: {e}")

    if not docs:
        logger.error("🚫 No documents to process. Aborting.")
        return None

    logger.info(f"📄 Loaded {len(docs)} document(s)")

    # Step 3: Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    logger.info(f"✂️ Split into {len(texts)} chunks")
    logger.info(f"🧪 Sample chunk: {texts[0].page_content[:150]}...")

    # Step 4: Embed and store
    embedder = get_embedder()

    if overwrite or not os.path.exists(FAISS_INDEX_PATH):
        # Overwrite mode: create new store
        store = FAISS.from_documents(texts, embedder)
        store.save_local(FAISS_INDEX_PATH)
        logger.info("💾 New FAISS index created and saved")
    else:
        # Append mode: load and extend
        store = FAISS.load_local(FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True)
        store.add_documents(texts)
        store.save_local(FAISS_INDEX_PATH)
        logger.info("📌 New documents added to existing FAISS index")

    logger.info("✅✅ Vector store ready")
    return store
