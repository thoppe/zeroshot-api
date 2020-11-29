from fastapi import FastAPI
from pydantic import conlist

from data_models import SingleQuery
from utils import compute_with_cache
from utils import chunks

from utils import redis_instance

app = FastAPI()
__version__ = "0.0.1"


@app.get("/")
def read_root():
    import sys

    module_list = ['transformers', 'tokenizers', 'redis', 'torch', 'numpy']
    versions = {
        "Zero-Shot-API-version": __version__,
    }

    for key in module_list:
        versions[key] = sys.modules[key].__version__

    info = {
        "python_versions" : versions,
        "model_name" : utils.model_name,
        "device" : utils.device,
    }

    return versions

@app.get("/flush_cache")
def flush_cache():
    redis_instance.flushdb()


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
