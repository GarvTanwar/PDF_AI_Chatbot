from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.load_vectorstore import load_vectorstore
from modules.llm import getLlmChain
from modules.query_handlers import query_chain
from modules.logger import logger

app = FastAPI(title="PdfAIBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request:Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled Exception")
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post('/upload_pdfs/')
async def upload_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"Recieved {len(files)} files")
        load_vectorstore(files)
        logger.info("documents added to chroma")
        return {"message" : "Files are processed and vectorstore is updated"}
    except Exception as e:
        logger.exception("Error during PDF Upload")
        return JSONResponse(status_code=500, content={"error":str(e)})

@app.post('/ask/')
async def ask_questions(question:str=Form(...)):
    try:
        logger.info(f"User Query: {question}")
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from modules.load_vectorstore import PERSIST_DIR

        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L12-v2")
        )
        chain = getLlmChain(vectorstore)
        result = query_chain(chain, question)
        logger.info("Query Successfull")
        
        return result
    except Exception as e:
        logger.exception("Error Processing Question")
        return JSONResponse(status_code=500, content={"error":str(e)})

@app.get('/test')
async def test():
    return {"message" : "Testing Successfully"}

