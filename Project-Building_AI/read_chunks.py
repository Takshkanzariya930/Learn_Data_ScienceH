import requests
import pandas as pd
import json
import os

def create_embedding(text_list):
    r = requests.post("http://127.0.0.1:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

chunk_id = 0
my_dict = []

for json_file in os.listdir("Project-Building_AI/jsons"):
    
    with open(f"Project-Building_AI/jsons/{json_file}") as f:
        content = json.load(f)
    
    print(f"Creating embeddings for {content[0]["number"]}_{content[0]["title"]}")
    
    embeddings = create_embedding([chunk["text"] for chunk in content])
    
    for i, chunk in enumerate(content):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id = chunk_id + 1
        my_dict.append(chunk)


df = pd.DataFrame.from_records(my_dict)
print(df)