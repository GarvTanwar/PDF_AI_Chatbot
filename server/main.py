from fastapi import FastAPI

app = FastAPI(title="PdfAIBot")

@app.get('/test')
async def test():
    return {"message" : "Testing Successfully"}