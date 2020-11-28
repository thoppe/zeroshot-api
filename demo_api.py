import requests
import json
from wasabi import msg
import pandas as pd

url = 'http://127.0.0.1:8000/zs'

params = {
    'hypothesis_template' : "I like to go {}",
    'candidate_labels' : ["shopping", "swimming"],    
    'sequences' : ["I have a new swimsuit.",
                   'Today is a new day for everyone',
                   "I have money and I want to spend it."]
}

r = requests.post(url, json = params)

if not r.ok:
    msg.fail(f"Bad request status code: {r.status_code}")

    if r.status_code == 422:
        print(json.dumps(r.json(),indent=2))
        exit()

df = pd.read_json(r.json())
print(df)

