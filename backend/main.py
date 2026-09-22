from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Tamilrootlabs API")

# Fetch the Ollama URL from Docker environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

# Define the data format we expect from the Windows UI
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "Tamilrootlabs Backend is Online!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Receives text from UI, sends to Brain, returns raw response."""
    
    # Payload for Ollama
    payload = {
        "model": "deepseek-r1:7b", # Our chosen Brain
        "prompt": request.prompt,
        "stream": False
    }
    
    try:
        # Send request to Ollama container
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        raw_llm_output = response.json().get("response", "")
        
        # LATER: We will add the ##_think_## Regex Sanitizer right here
        
        return {"tamil_response": raw_llm_output}
    except Exception as e:
        return {"error": str(e)}

