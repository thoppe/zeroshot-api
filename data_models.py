from pydantic import BaseModel

class SingleQuery(BaseModel):
    hypothesis: str
    label: str
    sequence: str
