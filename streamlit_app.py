import streamlit as st
import requests
import pandas as pd

zs_url = "http://127.0.0.1:8000"

st.title("Zero-shot API with caching")

model_hypothesis = st.text_input(
    "Input the hypothesis using a {} as a label placeholder.", "I like to go {}.",
)

model_labels = st.text_area(
    "Input the labels, one on each line.", "shopping\nswimming",
)

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


def extract_valid_textlines(block):
    return [x.strip() for x in block.split("\n") if x.strip()]


def get_api_info():
    r = requests.get(zs_url + "/")
    return r.json()


def infer(labes, sequences):
    params = {"hypothesis": model_hypothesis, "labels": labels, "sequences": sequences}

    r = requests.get(zs_url + "/infer", json=params)
    df = pd.read_json(r.json())
    return df


def reset_cache():
    requests.get(zs_url + "/flush_cache")


info = get_api_info()

st.sidebar.markdown(f"model name: {info['model_name']}")
st.sidebar.markdown(f"device: {info['device']}")
st.sidebar.markdown(f"cached inferences: {info['cached_items']}")
btn_reset = st.sidebar.button("Reset cache 💣")

if btn_reset:
    reset_cache()
    info = get_api_info()

labels = extract_valid_textlines(model_labels)
sequences = extract_valid_textlines(model_sequences)

results = infer(labels, sequences)
st.table(results)
