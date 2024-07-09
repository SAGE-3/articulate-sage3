import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
import os
import wave
import contextlib
import collections
# import webrtcvad
from pydub import AudioSegment
import whisper
from datetime import datetime

import time
from groq import Groq
import csv_llm
import openai
import requests
import command_models
from collections import deque

filename = "./logs/test.json"
filename2 = "./logs/test2.json"

# Check if the file exists, if not create it with an empty array
if not os.path.exists(filename):
    with open(filename, 'w') as file:
        json.dump([], file)

def write_to_file(data):
    # Read the existing content
    with open(filename, 'r') as file:
        array = json.load(file)
    
    # Add new data to the array
    array.append(data)
    
    # Write the updated array back to the file
    with open(filename, 'w') as file:
        json.dump(array, file, indent=4)
        
    return

# Check if the file exists, if not create it with an empty array
if not os.path.exists(filename2):
    with open(filename2, 'w') as file:
        json.dump([], file)

def write_to_file2(data):
    # Read the existing content
    with open(filename2, 'r') as file:
        array = json.load(file)
    
    # Add new data to the array
    array.append(data)
    
    # Write the updated array back to the file
    with open(filename2, 'w') as file:
        json.dump(array, file, indent=4)
        
    return

class TextRequest(BaseModel):
    prompt: str
    context: str
    chartContext: str
    
class NL2CodeBody(BaseModel):
    text_prompt: str
    user_id: str 

whisper_model = whisper.load_model("base.en")
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


client = openai.OpenAI(
    base_url = os.environ["OPENAI_API_BASE"], # "http://<Your api-server IP>:port"
    api_key  = os.environ["OPENAI_API_KEY"]
)
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
model_path = os.path.join(current_dir, 'command_models', 'saved_models', 'command_modelV2.pt')

command_model = command_models.UtteranceClassifier(model_path, 3072, 128, 3, client)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies/authorization headers
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global deque to store the last 5 utterances
last_utterances = deque(maxlen=5)

@app.post("./transcribeAndProcess")
async def transcribeAndProcess(file: UploadFile = File(...)):
    contents = await file.read()
    temp_file_path = f"/tmp/{uuid.uuid4()}.webm"

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(contents)

    audio_segment = AudioSegment.from_file(temp_file_path, format="webm")
    wav_file_path = temp_file_path.replace(".webm", ".wav")
    
    # Convert to mono and 16-bit PCM with a supported sample rate
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio_segment.export(wav_file_path, format="wav")

    # if not has_voice_activity(wav_file_path):
    #     return {"transcription": ""}

    result = whisper_model.transcribe(wav_file_path)
    os.remove(temp_file_path)
    os.remove(wav_file_path)
    
    

    return {"transcription": result["text"]}
    

@app.post("/processIfCommand")
async def process_if_command(request: TextRequest):
  user_prompt = request.prompt #"""show me car brands vs price, sort by decending prices"""
  prediction = command_model.predict(user_prompt)
    # Update the global deque with the new utterance
  last_utterances.append(user_prompt)
  print(last_utterances)
  return prediction

@app.post("/processCommand")
async def createChartOptions(request: TextRequest):   
  print(request)
  # required_fields(["prompt"], data)
  start =time.time()
  chart_context = request.chartContext

  client_groq = Groq(api_key = os.getenv('GROQ_API_KEY'))

  dataset = os.path.join(__location__,"datasets/common_vars.csv")

  llm_re = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1, "dataset": dataset})
  llm_base = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 0,"dataset": dataset})
  llm_transform = csv_llm.LLM(client, {"model": "gpt-4o-2024-05-13", "temperature": 0,"dataset": dataset} )
  llm = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 0, "dataset": dataset})
  conversational_context = request.context
  user_prompt = request.prompt #"""show me car brands vs price, sort by decending prices"""
  user_prompt_modified, user_prompt_reasoning = llm_re.prompt_reiterate(user_prompt, f"""For extra context, here is the current verbal context of the last few utterances: {" ".join(last_utterances)}. 
                                                 Here is is the current chart context of what charts were last created, selected, and iteracted by the user: {chart_context} """)

  print("****Selecting Stations****")

  stations, station_reasoning = llm_base.prompt_select_stations(user_prompt_modified)
  # print(stations)
  # print(station_reasoning)
  print()
  
  if len(stations) == 0:
    return {}
  
  print("****Selecting Dates****")
  # print("****Starting Extracting Stations****")
  dates, dates_reasoning = llm_base.prompt_select_dates(user_prompt_modified)
  # print(dates)
  # print(dates_reasoning)
  print()
  
  if len(stations) == 0:
    return {}
  
  station_info = {}
  for idx,id in enumerate(stations):
      with open(f'./datasets/stationVariables/stationVariables.json') as f:
        variables = json.load(f)
      available_variable_names = []
      available_variable_ids = []
      for var in variables:
          available_variable_names.append(var['var_name'].replace(",",""))
          available_variable_ids.append(var['var_id'])
      available_variable_names.append("Date")
      available_variable_ids.append('Date')
      station_info[id] = {'available_variable_names': available_variable_names, 'available_variable_ids': available_variable_ids}
  station_chart_info = {}
  available_variable_names = station_info[id]['available_variable_names']
  available_variable_ids = station_info[id]['available_variable_ids']
  print("****Selecting Attributes****")
  print()
  
  if len(stations) == 0:
    return {}
  
  chosen_attributes_names, attrib_reasoning = llm_base.prompt_attributes(user_prompt_modified, available_variable_names)
  chosen_attribute_ids = []
  for attr in chosen_attributes_names:
      # print(attr, "--------------------", available_variable_names)
      # Check if the attribute exists in the available variables
      if attr in available_variable_names:
          index = available_variable_names.index(attr)
          # print(index)
          # Check if index is valid
          if index != -1:
              chosen_attribute_ids.append(available_variable_ids[index])
          else:
              break
      else:
          break
  print("****Selecting Transformations****", chosen_attribute_ids)
  transformations, trans_reasoning = llm_transform.prompt_transformations(user_prompt_modified, chosen_attribute_ids)
  chartType, chart_frequencies, chart_reasoning, chart_scope = llm.prompt_charts_via_chart_info(user_prompt_modified, chosen_attributes_names)
  for id in station_info.keys():
    station_chart_info[id] = {'attributes': chosen_attribute_ids, 'transformations': transformations, 'chartType': chartType, 'available_attribute_info': station_info[id], 'dates': dates} #TODO check if values exist
  # print(f"************Generated a {station_chart_info}**************")
  
  if len(stations) == 0:
    return {}
  
  end = time.time()
  
  summarized_response = llm_transform.prompt_summarize_reasoning(user_prompt, chart_reasoning, attrib_reasoning, dates_reasoning, station_reasoning, trans_reasoning)
  
  print("Total time elapsed:", end-start)
  
  print()
  
  
  chartInformation = {
    "userPrompt": user_prompt,
    "chartContext": chart_context,
    "chartType": chartType,
    "chartReasoning": chart_reasoning,
    "conversationalContext": conversational_context,
    "userPromptModified": user_prompt_modified,
    "userPromptReasoning": user_prompt_reasoning,
    "stationChosen": stations, 
    "stationReasoning": station_reasoning,
    "dates": dates,
    "datesReasoning": dates_reasoning,
    "attributes": chosen_attributes_names,
    "attributeReasoning": attrib_reasoning,
    "transformations": transformations,
    "transformationReasoning": trans_reasoning,
    "totalElapsedTime": end-start,
    "summarizedResponse": summarized_response,
    "date": datetime.now()
    }


  write_to_file(json.dumps(chartInformation, default=str))
  
  return {
      'station_chart_info': station_chart_info,
          "debug": chartInformation,
          }
  




@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    temp_file_path = f"/tmp/{uuid.uuid4()}.webm"

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(contents)

    audio_segment = AudioSegment.from_file(temp_file_path, format="webm")
    wav_file_path = temp_file_path.replace(".webm", ".wav")
    
    # Convert to mono and 16-bit PCM with a supported sample rate
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio_segment.export(wav_file_path, format="wav")

    # if not has_voice_activity(wav_file_path):
    #     return {"transcription": ""}

    result = whisper_model.transcribe(wav_file_path)
    os.remove(temp_file_path)
    os.remove(wav_file_path)
    write_to_file2(json.dumps({'utterace': result['text'], 'time': datetime.now()}, default=str))
    return {"transcription": result["text"]}

def read_wave(path):
    """Reads a .wav file."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

# def has_voice_activity(wav_file_path):
#     vad = webrtcvad.Vad(3)
#     audio, sample_rate = read_wave(wav_file_path)
#     frames = frame_generator(30, audio, sample_rate)
#     frames = list(frames)

#     return any(vad.is_speech(frame.bytes, sample_rate) for frame in frames)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
