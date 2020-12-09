import streamlit as st
import pandas as pd
import json
import itertools
from configparser import ConfigParser
from explainer import compute_correlations

config = ConfigParser()
config.read("config.ini")

n_max_upload_rows = 80

zs_url = config.get("streamlit", "zs_url")
n_minibatch = int(config.get("streamlit", "n_minibatch"))

st.title("Zero-shot API explainer")

candidate_labels = st.text_area(
    "Input the labels",
    "working out\n"
    "eating\n"
)
candidate_labels 


foobar = '''

model_hypotheses = st.text_area(
    "Input the hypotheses, one on each line. To add a label suffix it with a semicolon.",
    "I like to go shopping. ; shop \n"
    "I like to go swimming. ; swim \n"
    "The moon is bright today.",
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


def infer(hypotheses, sequences, labels):
    params = {"hypotheses": hypotheses, "sequences": sequences}

    results = {}
    progress_bar = st.progress(0)
    n_total_items = len(hypotheses) * len(sequences)
    n_current = 0

    for hyp in hypotheses:
        results[hyp] = {}

        for chunk in chunks(sequences, n_minibatch):
            params = {
                "hypotheses": [hyp],
                "sequences": chunk,
            }

            r = requests.get(zs_url + "/infer", json=params)
            r = json.loads(r.json())

            results[hyp].update(r[hyp])
            n_current += len(chunk)

            progress_bar.progress(n_current / n_total_items)

    progress_bar.empty()

    df = pd.DataFrame(results)

    # Make sure output matches input order
    df = df[hypotheses].loc[sequences]

    # Apply the labels if provided
    df = df.rename(columns={k: (v if v else k) for k, v in zip(hypotheses, labels)})

    return df


def reset_cache():
    requests.get(zs_url + "/flushdb")


info = get_api_info()

msg = f"""
model name: {info['model_name']}
device: {info['device']}
cached inferences: {info['cached_items']}
""".strip()
st.sidebar.text(msg)


f_dataset = st.sidebar.file_uploader(
    "Upload a CSV. Target column must be labeled 'text'. "
    f"Data is limited to {n_max_upload_rows} rows."
)

# Load the data from file if it exists
if f_dataset is not None:
    # Example of loading a small dataset
    df = pd.read_csv(f_dataset)
    df = df[["text"]][:n_max_upload_rows]
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

btn_reset = st.sidebar.button("💣 Reset cache")

if btn_reset:
    reset_cache()


# Extract the hypotheses/labels from the input, apply a numerical
# label if missing from the input
hypotheses_labels = extract_valid_textlines(model_hypotheses)
hypotheses = [line.split(";")[0].strip() for line in hypotheses_labels]

labels = [line.split(";")[1:] for line in hypotheses_labels]
labels = [label[0] if label else None for label in labels]

results = infer(hypotheses, sequences, labels)
tableviz = results.style.background_gradient(cmap="Blues", axis=None).format("{:0.3f}")

st.table(tableviz)

st.sidebar.markdown(
    "🌱 [Source](https://github.com/thoppe/zeroshot-api) by [@metasemantic](https://twitter.com/metasemantic?lang=en)"
)
'''
