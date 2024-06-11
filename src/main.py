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
import webrtcvad
from pydub import AudioSegment
import whisper

import time
from groq import Groq
import csv_llm
import openai
import requests

class TextRequest(BaseModel):
    prompt: str
    context: str
    
class NL2CodeBody(BaseModel):
    text_prompt: str
    user_id: str 

whisper_model = whisper.load_model("base.en")
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

client = openai.OpenAI(
    base_url = os.environ["OPENAI_API_BASE"], # "http://<Your api-server IP>:port"
    api_key  = os.environ["OPENAI_API_KEY"]
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies/authorization headers
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

    if not has_voice_activity(wav_file_path):
        return {"transcription": ""}

    result = whisper_model.transcribe(wav_file_path)
    os.remove(temp_file_path)
    os.remove(wav_file_path)
    
    

    return {"transcription": result["text"]}

@app.post("/processText")
async def process_text(request: TextRequest):
    print(request)
    # required_fields(["prompt"], data)
    start =time.time()

    client_groq = Groq(api_key = os.getenv('GROQ_API_KEY'))

    dataset = os.path.join(__location__,"datasets/hcdpDataReduced.csv")
    few_shot = os.path.join(__location__, "data/finetune_train_articulate.xlsx")


    # llm_re = csv_llm.LLM(client, {"model": "gpt-3.5-turbo-0125", "temperature": 1})
    # llm_base = csv_llm.LLM(client, {"model": "gpt-3.5-turbo-0125", "temperature": 0}, context_path=dataset, iter_self_reflection=4)

    llm_re = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1})
    llm_base = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 0}, context_path=dataset)
    llm_transform = csv_llm.LLM(client, {"model": "gpt-4-turbo-preview", "temperature": 0}, context_path=dataset)
    llm = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1}, context_path=dataset)

    user_prompt = request.prompt #"""show me car brands vs price, sort by decending prices"""
    conversational_context = request.context
    
    print("****Starting Reiteration****")

    user_prompt_modified = llm_re.prompt_reiterate(user_prompt, conversational_context)
    print(user_prompt_modified)       
    print()
    
    print("****Starting Extracting Stations****")
    stations, station_reasoning = llm_base.prompt_select_stations(user_prompt_modified)
    print(stations)
    print(station_reasoning)
    
    
    
    print("****Starting for loop****")
    print()
    station_info = {}
    headers = {"Authorization": "Bearer 71c5efcd8cfe303f2795e51f01d19c6"}
    for idx,id in enumerate(stations):
        if idx >= 3:
            break
        r = requests.get(f"https://api.hcdp.ikewai.org/mesonet/getVariables?station_id={id}", headers=headers)
        variables = r.json()
        available_variable_names = []
        available_variable_ids = []
        for var in variables:
            available_variable_names.append(var['var_name'].replace(",",""))
            available_variable_ids.append(var['var_id'])
        available_variable_names.append("Date")
        available_variable_ids.append('Date')
        station_info[id] = {'available_variable_names': available_variable_names, 'available_variable_ids': available_variable_ids}
    station_chart_info = {}
    for id in station_info.keys():
        print("****Starting Attribute Selection****")
        print()
        print('available variable names ****', station_info[id]['available_variable_names'])
        chosen_attributes_names, attrib_reasoning = llm_base.prompt_attributes(user_prompt_modified, station_info[id]['available_variable_names'])
        chosen_attribute_ids = []
        for attr in chosen_attributes_names:
            print(attr, "--------------------", available_variable_names)
            # Check if the attribute exists in the available variables
            if attr in available_variable_names:
                index = station_info[id]['available_variable_names'].index(attr)
                print(index)
                # Check if index is valid
                if index != -1:
                    chosen_attribute_ids.append(station_info[id]['available_variable_ids'][index])
                else:
                    break
            else:
                break
        print(chosen_attribute_ids)
        print(attrib_reasoning)
        print()
        print("****Starting Transformation Selection****")
        print()
        transformations, trans_reasoning = llm_transform.prompt_transformations(user_prompt_modified, chosen_attributes_names)
        print(trans_reasoning)
        print("Transformations:", transformations)
        print()
        print("****Starting Chart Selection****")
        print()
        chartType, chart_frequencies, chart_reasoning, chart_scope = llm.prompt_charts_via_chart_info(user_prompt_modified, chosen_attributes_names)
        print(chart_reasoning)
        print(chartType)
        print(chart_frequencies)
        print(chart_scope)
        station_chart_info[id] = {'attributes': chosen_attribute_ids, 'transformations': transformations, 'chartType': chartType, 'available_attribute_info': station_info[id]} #TODO check if values exist
    # print("****Starting Extracting Date Ranges****")
    # stations, station_reasoning = llm_transform.prompt_select_date_range(user_prompt_modified)
    # print(stations)
    print(station_reasoning)





    # print(trans_reasoning)
    # print(transformations)
    end = time.time()
    
    print("Total time elapsed:", end-start)
    
    # attributes.append("Date")
    return {
        'station_chart_info': station_chart_info,
            # "attributes": {attribute: {"Data Type": column["Data Type"]} for attribute in attributes for column in csv_analysis if attribute == column["Column Name"]}, 
            # "csv_uuid": csv_uuid_ext.split(".")[0], 
            "debug": {
                "context": conversational_context,
                "query": user_prompt,
                "reiteration": user_prompt_modified, 
                "time": end-start,
                # "attributes": attrib_reasoning, 
                # "transformations": transformations,
                # "charts": chart_reasoning,
                # "charts_frequency": chart_frequencies,
                # "charts_scope": chart_scope,
                },
            # "analysis": csv_analysis
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

    if not has_voice_activity(wav_file_path):
        return {"transcription": ""}

    result = whisper_model.transcribe(wav_file_path)
    os.remove(temp_file_path)
    os.remove(wav_file_path)

    return {"transcription": result["text"]}

def read_wave(path):
    """Reads a .wav file."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def has_voice_activity(wav_file_path):
    vad = webrtcvad.Vad(3)
    audio, sample_rate = read_wave(wav_file_path)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)

    return any(vad.is_speech(frame.bytes, sample_rate) for frame in frames)

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
