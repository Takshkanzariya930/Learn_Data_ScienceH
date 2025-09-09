import whisper
import json
import os

model = whisper.load_model("large-v2", device="cuda")

audio = "09_Build a full stack project in Django for beginners.mp3"

number = audio.split("_")[0]
title = audio.split("_")[1][:-4]

result = model.transcribe(audio=f"Project-Building_AI/audios/{audio}", language="hi", task="translate", word_timestamps=False)

chunk = []
for segment in result["segments"]:
    chunk.append({"number" : number, "title" : title, "start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})
    
chunk_with_metadata = {"chunk" : chunk, "text" : result["text"]}
    
with open(f"Project-Building_AI/jsons/{audio[:-4]}.json", "w") as f:
    json.dump(chunk, f)