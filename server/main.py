from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.load_vectorstore import load_vectorstore, get_embedder
from modules.llm import getLlmChain
from modules.query_handlers import query_chain
from modules.logger import logger
import asyncio
from langchain_community.vectorstores import FAISS  # ✅ Corrected import

# Set your FAISS index path
FAISS_INDEX_PATH = "./faiss_index"

app = FastAPI(title="PdfAIBot")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Prevent concurrent vectorstore access
vector_lock = asyncio.Lock()

# Global error handler
@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled Exception")
        return JSONResponse(status_code=500, content={"error": str(exc)})

# PDF Upload Endpoint
@app.post('/upload_pdfs/')
async def upload_pdfs(files: List[UploadFile] = File(...)):
    async with vector_lock:
        try:
            logger.info(f"Recieved {len(files)} files")
            logger.info("documents added to vector store")

            in_memory_files = [(f.filename, await f.read()) for f in files]
            logger.info("Loading multiple files into memory")
            load_vectorstore(in_memory_files, overwrite=False)  # You can change to False if needed

            return {"message": "Files are processed and vectorstore is updated"}
        except Exception as e:
            logger.exception("Error during PDF Upload")
            return JSONResponse(status_code=500, content={"error": str(e)})

# Ask Questions Endpoint
@app.post('/ask/')
async def ask_questions(question: str = Form(...)):
    try:
        logger.info(f"User Query: {question}")
        embedder = get_embedder()

        # ✅ Fixed: enable safe loading
        store = FAISS.load_local(
            folder_path=FAISS_INDEX_PATH,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )

        chain = getLlmChain(store)
        result = query_chain(chain, question)
        logger.info("Query processed successfully")

        return result
    except Exception as e:
        logger.exception("Error Processing Question")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Health check
@app.get('/test')
async def test():
    return {"message": "Testing Successfully"}
