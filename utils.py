from typing import List, Union
import torch
import numpy as np

from data_models import SingleQuery
import redis
from configparser import ConfigParser


def load_NLP_tokenizer(model_name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def load_NLP_model(model_name, device):
    from transformers import AutoModelForSequenceClassification
    from transformers import logging

    # BART model always complains that we aren't loading everything
    # this is OK for this purpopse and we can ignore it
    logging.set_verbosity_error()

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return model


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def tokenize(Q: List[SingleQuery]):
    """
    Takes a list of queries and encodes them as pytorch inputs ready for inference.
    """

    # Entailment questions are encoded as (sequence, hypothesis)
    sequence_pairs = [(q.sequence, q.hypothesis) for q in Q]

    # Tokenize the results to prep for model
    inputs = tokenizer(
        sequence_pairs,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        max_length=max_token_length,
        truncation="only_first",
    )

    # Map the results onto the device (needed for CUDA)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    return inputs


def redis_encode(q: Union[SingleQuery, str]) -> str:
    """
    Encodes the zero-shot question into a fully formed hypothesis by applying 
    the label. If input is already a string, ignore it and pass it on.
    """

    if isinstance(q, str):
        return q

    return f"{q.hypothesis} : {q.sequence}"


def model_compute(Q: List[SingleQuery]) -> List[float]:
    """
    Uses loaded pytorch model to compute the entailments.
    """

    tokens = tokenize(Q)

    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
        )

    # Outputs are [Logits, Attention Heads, ...]
    logits = outputs[0]

    # Keep only the entailment and contradiction logits
    contradiction_id = 0
    entailment_id = 2
    logits = logits[..., [contradiction_id, entailment_id]]

    # Detach the logits from the computation graph
    logits = logits.detach().cpu().numpy()

    # Softmax over remaining logits for each sequence
    scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)

    del outputs
    del tokens
    torch.cuda.empty_cache()

    # Return the value entailment
    return scores[..., -1]


def compute_with_cache(Q: Union[SingleQuery, List[SingleQuery]]):

    batch_size = infer_arguments["batch_size"]

    # Force input to be a list
    if not isinstance(Q, List):
        Q = list(Q)

    # Encode the hypothesis+label to a redis key
    keys = np.array(list(map(redis_encode, Q)))

    # Determine what we've already computed
    R = redis_instance
    known_results = np.array(R.mget(keys), dtype=float)

    # Mark all missing results
    idx = np.isnan(known_results)

    # Compute the missing results if needed
    if idx.any():

        # Chunk the results on batch_size
        for batch_idx in chunks(np.where(idx)[0], infer_arguments["batch_size"]):
            print(f"Infering {len(batch_idx)} values")

            # Get tokens for remaining results
            remaining_Q = np.array(Q)[batch_idx]
            scores = model_compute(remaining_Q)

            # Cache the results
            R.mset({k: float(v) for k, v in zip(keys[batch_idx], scores)})

            known_results[batch_idx] = scores

    return known_results


##############################################################################
# Autoload model and redis cache
##############################################################################


redis_instance = redis.Redis()

config = ConfigParser()
config.read("config.ini")

model_name = config.get("model", "model_name")
device = config.get("model", "device")
max_token_length = int(config.get("api", "max_token_length"))

tokenizer = load_NLP_tokenizer(model_name)
model = load_NLP_model(model_name, device)

# Default arguments for inference

infer_arguments = {
    "is_unidecode": True,
    "batch_size": 8,
    "max_token_length": 512,
    "model_name": "facebook/bart-large-mnli",
}

# Update them from the config
if "api" in config:
    infer_arguments.update(dict(config["api"]))


# Force typing
infer_arguments["batch_size"] = int(infer_arguments["batch_size"])
infer_arguments["max_token_length"] = int(infer_arguments["max_token_length"])
