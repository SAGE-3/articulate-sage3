import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

import uuid
import os
import whisper

class NL2CodeBody(BaseModel):
    text_prompt: str
    user_id: str 

whisper_model = whisper.load_model("base.en")
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies/authorization headers
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print('fuck')
    contents = await file.read()
    temp_file_path = f"/tmp/{uuid.uuid4()}.webm"

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(contents)

    result = whisper_model.transcribe(temp_file_path)
    os.remove(temp_file_path)

    return {"transcription": result["text"]}
 
if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
    
  