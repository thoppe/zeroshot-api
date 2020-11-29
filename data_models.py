from pydantic import BaseModel
from typing import List, Union


class SingleQuery(BaseModel):
    hypothesis: str
    label: str
    sequence: str


class WebQuery(BaseModel):
    hypothesis: str
    labels: Union[str, List[str]]
    sequences: Union[str, List[str]]
