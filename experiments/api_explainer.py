# To be used when ready


@app.get("/explain")
def explain(Q: ExplainerQuery):
    """
    Computes the correlation between each word of the sequence and
    the "label" within the hypothesis. The correlation is over all the 
    concatenated encoder attention heads and is tokenwise.

    If the label contains multiple tokens, the average is taken.

    Returns a dataframe where each row is a word (BPE tokens combined)
    the first column is "word" with the text representation and each additional
    column is the label and the correlation of that label against the input
    sequence token.
    """

    df, scores = explainer.compute_correlations(Q)
    df = explainer.compress_frame(df)

    return {
        "correlations": df.to_json(),
        "scores": scores.tolist(),
    }
