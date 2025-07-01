from fastapi import FastAPI
import langchain

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": langchain.__version__}