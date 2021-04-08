import pandas as pd
import numpy as np
import spacy
from unidecode import unidecode
from tqdm import tqdm
import gensim.corpora as corpora

df = pd.read_csv("docs/lonely_post_features_tfidf_256.csv")
text = df["text"].str[:2000]

df = df[:50000]


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatization(text):
    tags = set(["NOUN", "ADJ", "VERB", "ADV"])
    doc = nlp(unidecode(text))
    return [token.lemma_ for token in doc if token.pos_ in tags]


docs = [lemmatization(text) for text in tqdm(df["text"], total=len(df))]

from gensim.corpora import Dictionary

id2word = corpora.Dictionary(docs)
id2word.filter_extremes(no_below=5, no_above=0.25)
corpus = [id2word.doc2bow(doc) for doc in docs]

from gensim.models.ldamodel import LdaModel

bow = [id2word.doc2bow(text) for text in docs]
lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=10)

print(lda)
for idx, topic in lda.show_topics(num_words=5, formatted=False):
    print(idx, "; ".join([x[0] for x in topic]))

exit()
# orpus = [

X = np.zeros(shape=(len(df), len(id2word)), dtype=int)

for i, text in enumerate(docs):
    for j in id2word.doc2idx(text):
        if j >= 0:
            X[i, j] += 1

import lda

model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_

n_top_words = 8
for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(vocab)[np.argsort(topic_dist)][: -(n_top_words + 1) : -1]
    print("Topic {}: {}".format(i, " ".join(topic_words)))
