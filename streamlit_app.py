import streamlit as st
import requests
import pandas as pd
import json
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")

zs_url = config.get("streamlit", "zs_url")
n_minibatch = int(config.get("streamlit", "n_minibatch"))

st.title("Zero-shot API with caching")

model_hypothesis = st.text_input(
    "Input the hypothesis using a {} as a label placeholder.", "I like to go {}.",
)

model_labels = st.text_area(
    "Input the labels, one on each line.", "shopping\nswimming",
)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def extract_valid_textlines(block):
    return [x.strip() for x in block.split("\n") if x.strip()]


def get_api_info():
    r = requests.get(zs_url + "/")
    return r.json()


def infer(labels, sequences):
    params = {"hypothesis": model_hypothesis, "labels": labels, "sequences": sequences}

    results = {}
    progress_bar = st.progress(0)
    n_total_items = len(labels) * len(sequences)
    n_current = 0

    for label in labels:
        results[label] = {}

        for chunk in chunks(sequences, n_minibatch):
            params = {
                "hypothesis": model_hypothesis,
                "labels": [label],
                "sequences": chunk,
            }
            r = requests.get(zs_url + "/infer", json=params)
            r = json.loads(r.json())

            results[label].update(r[label])
            n_current += len(chunk)

            progress_bar.progress(n_current / n_total_items)

    progress_bar.empty()
    df = pd.DataFrame(results)
    return df


def reset_cache():
    requests.get(zs_url + "/flush_cache")


info = get_api_info()

st.sidebar.markdown(f"model name: {info['model_name']}")
st.sidebar.markdown(f"device: {info['device']}")
st.sidebar.markdown(f"cached inferences: {info['cached_items']}")


f_dataset = st.sidebar.file_uploader(
    "Upload a CSV. Target column must be labeled 'text'"
)

# Load the data from file if it exists
if f_dataset is not None:
    # Example of loading a small dataset
    df = pd.read_csv(f_dataset)
    df = df[["text"]][:80]
    sequences = df["text"].fillna("").tolist()

# Otherwise take direct user input
else:
    model_sequences = st.text_area(
        "Input the sequences to infer, one each line.",
        "\n".join(
            [
                "I have a new swimsuit.",
                "Today is a new day for everyone.",
                "I have money and I want to spend it.",
            ]
        ),
    )

    sequences = extract_valid_textlines(model_sequences)

btn_reset = st.sidebar.button("Reset cache 💣")

if btn_reset:
    reset_cache()

labels = extract_valid_textlines(model_labels)

results = infer(labels, sequences)

tableviz = results.style.background_gradient(cmap="Blues").format("{:0.3f}")
# st.dataframe(tableviz)
st.table(tableviz)
