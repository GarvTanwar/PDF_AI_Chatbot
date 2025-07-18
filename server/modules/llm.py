import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def getLlmChain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        api_key = GEMINI_API_KEY,
        model = "gemini-1.5-flash"
    )
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 15})
    return RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True
    )