import streamlit as st
import pandas as pd
import json
import requests
from configparser import ConfigParser

import numpy as np
from annotation_component import annotated_text

config = ConfigParser()
config.read("config.ini")

n_max_upload_rows = 80

zs_url = config.get("streamlit", "zs_url")
n_minibatch = int(config.get("streamlit", "n_minibatch"))

st.title("Zero-shot API explainer")
st.write("Sample text")

sequence = st.text_input(
    "Input the sequence:",
    "I'm healthy, but stressed, kinda depressed but just started exercising and eating more veggies again.",
)

hypothesis_template = st.text_input(
    "Input the hypothesis template:", "The respondant copes by {}"
)

labels = st.text_area("Input the labels:", "working out\n" "eating\n")


def extract_valid_textlines(block):
    return [x.strip() for x in block.split("\n") if x.strip()]


def explain(hypothesis_template, sequence, labels):
    params = {
        "hypothesis_template": hypothesis_template,
        "sequence": sequence,
        "labels": labels,
    }

    r = requests.get(zs_url + "/explain", json=params)
    df = pd.DataFrame(r.json())

    return df


# Clean up the text in the labels
labels = extract_valid_textlines(labels)

df = explain(hypothesis_template, sequence, labels)

for label in labels:
    st.markdown(f"## {label}")

    annotated_text(df, "word", label, cmap_name="PuRd")


tableviz = df.style.background_gradient(cmap="PuRd", axis=None).format(
    "{:0.3f}", subset=labels
)
st.table(tableviz)


leftover_code = """
results = infer(hypotheses, sequences, labels)
tableviz = results.style.background_gradient(cmap="Blues", axis=None).format("{:0.3f}")

st.table(tableviz)

st.sidebar.markdown(
    "🌱 [Source](https://github.com/thoppe/zeroshot-api) by [@metasemantic](https://twitter.com/metasemantic?lang=en)"
)
"""
