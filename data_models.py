from pydantic import BaseModel
from typing import List, Union


class SingleQuery(BaseModel):
    hypothesis: str
    sequence: str


class MultiQuery(BaseModel):
    hypotheses: Union[str, List[str]]
    sequences: Union[str, List[str]]


class ExplainerQuery(BaseModel):
    hypothesis_template: str
    labels: List[str]
    sequence: str
