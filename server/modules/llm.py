import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def getLlmChain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        api_key = GEMINI_API_KEY,
        model_name = "gemini-pro"
    )
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})
    return RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True
    )