from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Tamilrootlabs API")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "Tamilrootlabs Backend is Online!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": request.prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        raw_llm_output = response.json().get("response", "")
        return {"tamil_response": raw_llm_output}
    except Exception as e:
        return {"error": str(e)}

