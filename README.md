# Zeroshot API
_Huggingface [zero-shot](https://joeddav.github.io/blog/2020/05/29/ZSL.html) module using fastapi and caching_

Install the requirements

    pip install -r requirements.txt

Start the service

    uvicorn interface:app


Test the service:

```python
import requests
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
df = pd.read_json(r.json())
print(df)
```

This gives

```
   shopping  swimming                              sequence
0  0.809227  0.969889                I have a new swimsuit.
1  0.109183  0.170154       Today is a new day for everyone
2  0.972724  0.384538  I have money and I want to spend it.
```
