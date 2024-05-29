import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

class NL2CodeBody(BaseModel):
    text_prompt: str
    user_id: str 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies/authorization headers
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/test")
def nl2code():

    print('hiii')
    data = json.dumps({"reply": "hello 2"})
    return data
 
if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)