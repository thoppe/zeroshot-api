from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

import torch
import pandas as pd

class ZS_Query(BaseModel):
    hypothesis_template : str
    candidate_labels : conlist(str, min_items=1)
    sequences : conlist(str, min_items=1)

app = FastAPI()

def load_model():
    from transformers import pipeline
    
    model_name = "facebook/bart-large-mnli"
    device_id = -1 # CPU only

    nlp = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device_id,
    )

    return nlp



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/zs")
def compute(q: ZS_Query):

    embedding = []
    
    for chunk in chunks(q.sequences, n_minibatch):
        
        with torch.no_grad():
            outputs = nlp(
                chunk,
                q.candidate_labels,
                multi_class=True,
                hypothesis_template=q.hypothesis_template,
            )
            

        for item in outputs:
            record = {}

            for label, score in zip(item["labels"], item["scores"]):
                record[label] = score
            
            embedding.append(record)

    
    dx = pd.DataFrame(embedding)[q.candidate_labels]
    dx['sequence'] = q.sequences

    return dx.to_json()


n_minibatch = 8
nlp = load_model()
print("Model loaded")
