import utils
import torch
import numpy as np
import pandas as pd

from data_models import SingleQuery
from scipy.spatial.distance import cdist


def token_chunks(s: str, s2=None, add_special_tokens=False):
    text = s if s2 is None else (s, s2)

    tokens = utils.tokenizer(
        [text],
        return_tensors="np",
        truncation="only_first",
        add_special_tokens=add_special_tokens,
    )

    return utils.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])


def measure_token_length(s: str):
    """
    Measures the length of an input sequence in terms of tokens
    """
    return len(token_chunks(s))


def compute_correlations(sequence, hypothesis_template, candidate_labels):
    Q = [
        SingleQuery(hypothesis=hypothesis_template.format(label), sequence=doc)
        for label in candidate_labels
    ]
    tokens = utils.tokenize(Q)

    # Run the forward pass, saving the encoder hidden states
    with torch.no_grad():
        outputs = utils.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    # Keep only the encoder hidden states
    layers = outputs["encoder_hidden_states"]

    # Remove from the GPU and convert to a numpy array
    # (layer, candidate_labels, token, embedding)
    state = np.array([layer.detach().cpu().numpy() for layer in layers])

    # Make the state easier to work with
    # (candidate_labels, token, layer, embedding)
    state = state.transpose(1, 2, 0, 3)

    # Maybe cutoff some of the bottom layers?
    # TO DO

    # Squish the layers and embeddings for correlation calculation
    # (candidate_labels, token, layer(+)embedding)
    state = state.reshape((*state.shape[:-2], -1))

    # Measure the length of the hypothesis and response
    hypothesis_prefix = hypothesis_template.split("{}")[0]
    n_hypothesis_prefix = measure_token_length(hypothesis_prefix)
    n_response = measure_token_length(doc)

    df = pd.DataFrame()

    for label, V in zip(candidate_labels, state):

        # Measure where each label via tokens starts and ends
        n_label = measure_token_length(label)
        n_label_start = n_response + n_hypothesis_prefix + 2
        n_label_end = n_label_start + n_label

        # (words, label_length)
        C = 1 - cdist(V, V[n_label_start:n_label_end], metric="correlation")

        # Add up over all label words
        C = np.sum(C, axis=1)

        # Normalize to the self-correlation
        C /= np.average(C[n_label_start:n_label_end])

        # Truncate to the response only
        C = C[1 : n_response + 1]

        df[label] = C

    df.insert(0, "words", token_chunks(doc))

    return df


if __name__ == "__main__":

    doc = "I'm healthy, but stressed, kinda depressed but just started exercising and eating more veggies again."
    hypothesis_template = "The respondant copes by {}."
    candidate_labels = ["working out", "eating"]

    df = compute_correlations(doc, hypothesis_template, candidate_labels)
    print(df)
