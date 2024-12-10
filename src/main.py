import os
from dotenv import load_dotenv
from fastapi import FastAPI
from src.rec_system.router import router as rec_router




# config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.include_router(rec_router)


@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}

