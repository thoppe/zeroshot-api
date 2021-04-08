# Zero-shot API
[Zero-shot](https://joeddav.github.io/blog/2020/05/29/ZSL.html) inference with huggingface [transformers](https://huggingface.co/) using [FastAPI](https://fastapi.tiangolo.com/) and caching with [Redis](https://github.com/andymccurdy/redis-py)

Install the requirements

    pip install -r requirements.txt

Start the service

    uvicorn api:app


Test the service:

```python
import requests
import pandas as pd

url = 'http://127.0.0.1:8000/infer'

params = {
    "hypotheses": ["I like to go swimming", "I like to go shopping",],
    "sequences": [
        "I have a new swimsuit.",
        "Today is a new day for everyone.",
        "I have money and I want to spend it.",
    ],
}

r = requests.get(base_url + "/infer", json=params)
df = pd.read_json(r.json())
print(df)
```

This gives

```

                               I like to go swimming  I like to go shopping
I have a new swimsuit.                      0.969889               0.809227
Today is a new day for everyone.            0.186691               0.162844
I have money and I want to spend it.        0.384539               0.972724
```

With the API running, you can experiment with a barebones streamlit interface:

    streamlit run streamlit_app_infer.py

![](docs/streamlit_example_infer.png)

----------------------------------------------------------------------------

_experimental work-in-progress currently not functional_

You can also experiment with the explainer:

    streamlit run streamlit_app_explainer.py

![](docs/streamlit_example_explain.png)


### Roadmap

+ Streamlit component explainer should match format of larger app
+ Incorporate MALLET LDA as preseed topics?