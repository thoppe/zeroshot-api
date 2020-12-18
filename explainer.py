import utils
import torch
import numpy as np
import pandas as pd

from data_models import SingleQuery, ExplainerQuery
from scipy.spatial.distance import cdist


def _token_chunks(s: str, s2=None, add_special_tokens=False):
    """
    Helper function to tokenize without special tokens and returning
    only a numpy array for speed.
    """

    text = s if s2 is None else (s, s2)

    tokens = utils.tokenizer(
        [text],
        return_tensors="np",
        truncation="only_first",
        add_special_tokens=add_special_tokens,
    )

    return utils.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])


def _measure_token_length(s: str):
    """
    Measures the length of an input sequence in terms of tokens
    """
    return len(_token_chunks(s))


def compute_correlations(query: ExplainerQuery):
    """
    Computes the correlation between each word of the sequence and
    the "label" within the hypothesis. The correlation is over all the 
    concatenated encoder attention heads and is tokenwise.

    If the label contains multiple tokens, the average is taken.

    Returns a dataframe and list:
       - The dataframe where each row is a token, the first column is
    "word" with the text representation of the token and each additional
    column is the label and the correlation of that label against the input
    sequence token.
       - The list are scores coming from each label
    """

    hypothesis_template = query.hypothesis_template
    candidate_labels = query.labels
    sequence = query.sequence

    Q = [
        SingleQuery(hypothesis=hypothesis_template.format(label), sequence=sequence)
        for label in candidate_labels
    ]
    tokens = utils.tokenize(Q)

    # Run a single forward pass, saving the encoder hidden states
    with torch.no_grad():
        outputs = utils.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    # Get the logits from the output
    logits = outputs["logits"].detach().cpu().numpy()

    # Keep only the entailment and contradiction logits
    contradiction_id = 0
    entailment_id = 2
    logits = logits[..., [contradiction_id, entailment_id]]

    # Softmax over remaining logits for each sequence
    scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    scores = scores[..., -1]

    # Computer correlation from encoder hidden states
    layers = outputs["encoder_hidden_states"]

    # Remove from the GPU and convert to a numpy array
    # (layer, candidate_labels, token, embedding)
    state = np.array([layer.detach().cpu().numpy() for layer in layers])

    # Make the state easier to work with
    # (candidate_labels, token, layer, embedding)
    state = state.transpose(1, 2, 0, 3)

    # Squish the layers and embeddings for correlation calculation
    # (candidate_labels, token, layer(+)embedding)
    state = state.reshape((*state.shape[:-2], -1))

    # Measure the length of the hypothesis and response
    hypothesis_prefix = hypothesis_template.split("{}")[0]
    n_hypothesis_prefix = _measure_token_length(hypothesis_prefix)
    n_response = _measure_token_length(sequence)

    df = pd.DataFrame()

    for label, V in zip(candidate_labels, state):

        # Measure where each label via tokens starts and ends
        n_label = _measure_token_length(label)
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

    df.insert(0, "word", _token_chunks(sequence))

    return df, scores


def compress_frame(df):
    """
    Cleans the dataframe from compute_correlations by combining byte-pair
    encodings (BPE) back into full words. This may fail if not using BART.
    """

    BPE_space_char = "Ġ"
    combine_chars = "'-"

    final_df = None

    for col in df.columns[1:]:
        whole_word, vals = [], []
        data = []

        for word, val in zip(df.word, df[col]):

            start_char = word[0]

            if (start_char == BPE_space_char) or (
                not start_char.isalpha() and start_char in combine_chars
            ):

                data.append(
                    {"word": "".join(whole_word), col: np.average(vals),}
                )
                whole_word, vals = [], []

            whole_word.append(word)
            vals.append(val)

        if whole_word:
            data.append(
                {"word": "".join(whole_word), col: np.average(vals),}
            )

        dx = pd.DataFrame(data)
        dx["word"] = dx.word.str.strip(BPE_space_char)

        if final_df is None:
            final_df = dx
        else:
            final_df[col] = dx[col]

    return final_df


if __name__ == "__main__":

    doc = "I'm healthy, but stressed, kinda depressed but just started exercising and eating more veggies again."
    hypothesis_template = "The respondant copes by {}."
    candidate_labels = ["working out", "eating", "sleeping"]

    Q = ExplainerQuery(
        hypothesis_template=hypothesis_template, labels=candidate_labels, sequence=doc
    )

    df, scores = compute_correlations(Q)
    #print(df)
    print(scores)
    df = compress_frame(df)
    #print(df)

    print(df.loc[[10, 12]])
