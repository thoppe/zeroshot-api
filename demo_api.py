import requests
import json
from wasabi import msg
import pandas as pd

base_url = "http://127.0.0.1:8000"
interface_url = "/mock_zs"

# Output the versions
r = requests.get(base_url + '/')
print(r.json())
exit()


params = {
    "hypothesis_template": "I like to go {}",
    "candidate_labels": ["shopping", "swimming"],
    "sequences": [
        "I have a new swimsuit.",
        "Today is a new day for everyone",
        "I have money and I want to spend it.",
    ],
}

# Clean the cache for testing
requests.get(base_url + "/flush_cache")





# First request two columns
r = requests.post(base_url + interface_url, json=params)

# Check if everything worked
if not r.ok:
    msg.fail(f"Bad request status code: {r.status_code}")

    if r.status_code == 422:
        print(json.dumps(r.json(), indent=2))
        exit()

# Check that we can convert to a dataframe
df = pd.read_json(r.json())
print(df)

# Now request a single additional column
# TO DO: Check that we can cache
params["candidate_labels"].append("dancing")
r = requests.post(base_url + interface_url, json=params)
df = pd.read_json(r.json())
print(df)
