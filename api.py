from fastapi import FastAPI
import unidecode

from data_models import SingleQuery, MultiQuery, ExplainerQuery
import utils

import pandas as pd
import numpy as np
import uvicorn  # Import needed for version number

app = FastAPI()
__version__ = "0.2.0"


# Unidecode is stupid and doesn't use __version__
unidecode.__version__ = f"{unidecode.version_info.major}.{unidecode.version_info.minor}.{unidecode.version_info.minor}"


@app.get("/")
def get_api_information():
    """
    Returns stats about the model and versions installed.
    """
    import sys

    module_list = [
        "transformers",
        "tokenizers",
        "redis",
        "fastapi",
        "uvicorn",
        "torch",
        "numpy",
        "unidecode",
        "pandas",
    ]
    versions = {
        "Zero-Shot-API-version": __version__,
    }

    for key in module_list:
        versions[key] = sys.modules[key].__version__

    info = {
        "python_versions": versions,
        "model_name": utils.model_name,
        "device": utils.device,
        "cached_items": dbsize(),
        "infer_arguments": utils.infer_arguments,
    }

    return info


@app.get("/flushdb")
def flushdb() -> None:
    """
    Flush the redis cache (useful for testing).
    """
    utils.redis_instance.flushdb()


@app.get("/configure")
def configure(key: str, value) -> None:
    """
    Sets a configuration argument.
    """
    assert key in utils.infer_arguments
    utils.infer_arguments[key] = value


@app.get("/dbsize")
def dbsize() -> int:
    """
    Returns the number of items cached.
    """
    return utils.redis_instance.dbsize()


def enforce_list(x):
    return (
        x if isinstance(x, list) else [x,]
    )


@app.get("/infer")
def infer(q: MultiQuery):
    """
    Run inference on a set of queries. Caching is not provided, user
    must do so or it can break the system.
    """

    # Cast sequences and labels a list
    sequences = enforce_list(q.sequences)
    hypotheses = enforce_list(q.hypotheses)

    # Encode if needed
    if utils.infer_arguments["is_unidecode"]:
        sequences = [unidecode.unidecode(x) for x in sequences]
        hypotheses = [unidecode.unidecode(x) for x in hypotheses]

    # Build a set of model queries
    Q = []
    for hyp in hypotheses:
        for seq in sequences:
            Q.append(SingleQuery(hypothesis=hyp, sequence=seq))

    v = utils.compute_with_cache(Q)

    # Reshape to match question labels
    v = np.reshape(v, (len(q.hypotheses), len(q.sequences)))
    df = pd.DataFrame(index=q.sequences, columns=q.hypotheses, data=v.T)

    # Return as a nice dataframe
    return df.to_json()


if __name__ == "__main__":
    import json

    # Flush the cache for testing
    flushdb()

    hypotheses = [
        "I like to go swimming",
        "I am very rich",
    ]
    sequences = [
        "I have a new swimsuit",
        "My dog bit another dog.",
    ]

    q = MultiQuery(hypotheses=hypotheses, sequences=sequences)

    info = get_api_information()
    configure("batch_size", 2)

    # Start another computation
    sequences.append("I won the lottery")
    q = MultiQuery(hypotheses=hypotheses, sequences=sequences)
    v = infer(q)
    print(v)

    doc = "I'm healthy, but stressed, kinda depressed but just started exercising and eating more veggies again."
    hypothesis_template = "The respondant copes by {}."
    candidate_labels = ["working out", "eating"]

    v = infer(q)
    print(v)

    """
    Q = ExplainerQuery(
        hypothesis_template=hypothesis_template, labels=candidate_labels, sequence=doc
    )

    results = explain(Q)
    print(results["correlations"])
    print(results["scores"])
    """
