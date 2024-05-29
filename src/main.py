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

@app.post("/nl2code")
def nl2code(body: NL2CodeBody):
    # Get text_prompt from body and assign to variable text_prompt
    text_prompt = body.text_prompt
    user_id = body.user_id

    # Create string for the print function
    print_function = f"print(\"{text_prompt}\")"
    data = json.dumps({"code": print_function})
    return data
 
if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)