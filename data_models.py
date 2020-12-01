from pydantic import BaseModel
from typing import List, Union, Optional


class SingleQuery(BaseModel):
    hypothesis: str
    sequence: str


class MultiQuery(BaseModel):
    hypotheses: Union[str, List[str]]
    sequences: Union[str, List[str]]
