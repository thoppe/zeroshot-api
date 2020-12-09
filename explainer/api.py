from fastapi import FastAPI

from data_models import ExplainerQuery
from explainer import compute_correlations


app = FastAPI()
__version__ = "0.1.0"

@app.get("/explain")
def explain(q: ExplainerQuery):
    """
    Explain a set of queries. Returns a dataframe where the first column
    is the collapsed set of words, and each subsequent column are the labels.
    The rows are words and the scores are the computation of the correlation
    across the encoder embedding.
    """

    df = compute_correlations(
        q.sequence,
        q.hypothesis_template,
        q.labels
    )
    print(df)
    
    
    return q

if __name__ == "__main__":
        
    doc = "I'm healthy, but stressed, kinda depressed but just started exercising and eating more veggies again."
    hypothesis_template = "The respondant copes by {}."
    candidate_labels = ["working out", "eating"]

    Q = ExplainerQuery(
        hypothesis_template=hypothesis_template,
        labels = candidate_labels,
        sequence = doc,
    )
    
    print(Q)
    explain(Q)
