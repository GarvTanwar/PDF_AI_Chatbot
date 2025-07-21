import requests
from config import API_KEY

def upload_pdfs_api(files):
    files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in files]

    return requests.post(f"{API_KEY}/upload_pdfs/", files=files_payload)

def ask_question(question):
    return requests.post(f"{API_KEY}/ask/", data={"question":question})