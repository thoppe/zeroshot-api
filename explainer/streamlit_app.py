import streamlit as st
import pandas as pd
import json
import requests
from configparser import ConfigParser

import numpy as np
from st_textmap_annotation import annotated_text

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

    r = requests.get(zs_url + "/explain", json=params).json()
    df = pd.DataFrame(json.loads(r["correlations"]))

    return df, r["scores"]


# Clean up the text in the labels
labels = extract_valid_textlines(labels)

df, scores = explain(hypothesis_template, sequence, labels)

for label, score in zip(labels, scores):
    st.markdown(f"## {label} ({score:0.3f})")

    annotated_text(df, "word", label, cmap_name="PuRd")


tableviz = df.style.background_gradient(cmap="PuRd", axis=None).format(
    "{:0.3f}", subset=labels
)
st.table(tableviz)
