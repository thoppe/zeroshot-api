from fastapi import FastAPI

from data_models import SingleQuery, MultiQuery
import utils

import pandas as pd
import numpy as np

app = FastAPI()
__version__ = "0.0.2"


@app.get("/")
def get_api_information():
    """
    Returns stats about the model and versions installed.
    """
    import sys

    module_list = ["transformers", "tokenizers", "redis", "torch", "numpy"]
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
    }

    return info


@app.get("/flushdb")
def flushdb() -> None:
    """
    Flush the redis cache (useful for testing).
    """
    utils.redis_instance.flushdb()


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
    v = infer(q)
    print(v)

    sequences.append("I won the lottery")
    q = MultiQuery(hypotheses=hypotheses, sequences=sequences)
    v = infer(q)
    print(v)
