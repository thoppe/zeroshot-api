from fastapi import FastAPI

from data_models import SingleQuery, WebQuery
from utils import chunks
import utils

import pandas as pd
import numpy as np

app = FastAPI()
__version__ = "0.0.1"


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
        "cached_items" : get_cache_size(),
    }

    return info


@app.get("/flush_cache")
def flush_cache():
    """
    Flushes the redis cache (useful for testing).
    """
    utils.redis_instance.flushdb()


@app.get("/count_cache")
def get_cache_size() -> int:
    """
    Returns the number of items cached.
    """
    return utils.redis_instance.dbsize()


def enforce_list(x):
    return (
        x if isinstance(x, list) else [x,]
    )


@app.get("/infer")
def infer(q: WebQuery):

    # Cast sequences and labels a list
    sequences = enforce_list(q.sequences)
    labels = enforce_list(q.labels)

    # Build a set of model queries
    Q = []
    for label in q.labels:
        for seq in q.sequences:
            Q.append(SingleQuery(hypothesis=q.hypothesis, label=label, sequence=seq))

    # TO DO: add batching
    v = utils.compute_with_cache(Q)

    # Reshape to match question labels
    v = np.reshape(v, (len(q.labels), len(q.sequences)))
    df = pd.DataFrame(index=q.sequences, columns=q.labels, data=v.T)

    # Return as a nice dataframe
    return df.to_json()


if __name__ == "__main__":

    # Flush the cache for testing
    flush_cache()

    q1 = SingleQuery(
        hypothesis="I like to go {}",
        label="swimming",
        sequence="I have a new swimsuit",
    )

    q2 = SingleQuery(
        hypothesis="I like to go {}", label="swimming", sequence="The bank needs money",
    )

    q3 = SingleQuery(
        hypothesis="I am oh so very {}", label="rich", sequence="I won the lottery.",
    )

    v = compute_with_cache([q1, q2])
    print(v)

    v = compute_with_cache([q1, q2, q3])
    print(v)
