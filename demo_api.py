import requests
import json
import pandas as pd
from contexttimer import Timer

base_url = "http://127.0.0.1:8000"

# Output the versions
r = requests.get(base_url + "/")
print(json.dumps(r.json(), indent=2))


params = {
    "hypothesis": "I like to go {}",
    "labels": ["shopping", "swimming"],
    "sequences": [
        "I have a new swimsuit.",
        "Today is a new day for everyone.",
        "I have money and I want to spend it.",
    ],
}

# Clean the cache for testing
requests.get(base_url + "/flush_cache")

with Timer() as timer0:
    r = requests.get(base_url + "/infer", json=params)

# Check if everything worked
if not r.ok:
    print(f"Bad request status code: {r.status_code}")

    if r.status_code == 422:
        print(json.dumps(r.json(), indent=2))
        exit()

# Check that we can convert to a dataframe
df = pd.read_json(r.json())
print(df)


with Timer() as timer1:
    for n in range(20):
        r = requests.get(base_url + "/infer", json=params)

# Now request a single additional column
params["labels"].append("dancing")

with Timer() as timer2:
    r = requests.get(base_url + "/infer", json=params)

df = pd.read_json(r.json())
print(df)

print("\nTiming results:")
print(f"First request    : {timer0.elapsed:0.3f} s")
print(f"Next 20 requests : {timer1.elapsed:0.3f} s")
print(f"Appened request  : {timer2.elapsed:0.3f} s")
