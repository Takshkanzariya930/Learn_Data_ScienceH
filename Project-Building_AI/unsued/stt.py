import whisper
import json

model = whisper.load_model("large-v2", device="cuda")

result = model.transcribe(audio="Project-Building_AI/audios/sample.mp3", language="hi", task="translate", word_timestamps=False)

chunk = []

for segment in result["segments"]:
    chunk.append({"start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})
    
print(chunk)

with open("Project-Building_AI/sample.json", "w") as f:
    json.dump(chunk, f)