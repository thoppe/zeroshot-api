import redis
import json
import numpy as np
import pandas as pd


class ZS_Redis_KeyStore(redis.Redis):
    def encode(self, item):
        """
        Expect item to have the following keys:
        hypothesis, label, sequence
        """

        assert "hypothesis" in item
        assert "label" in item
        assert "sequence" in item
        assert len(item.keys()) == 3

        key = item["hypothesis"].format(item["label"]) + " : " + item["sequence"]

        # return json.dumps(key)
        return key

    def encode_series(self, hypothesis, labels, sequences):

        keys = []
        for seq in sequences:
            for label in labels:
                keys.append(
                    {"hypothesis": hypothesis, "label": label, "sequence": seq,}
                )
        return keys

    def mget(self, keys):
        """
        Overridden mget, takes in a list of "keys"
        """

        keys = [self.encode(key) if isinstance(key, dict) else key for key in keys]

        vals = super().mget(keys)

        vals = [x if x is None else float(x) for x in vals]
        return vals

    def get(self, key):
        return self.mget([key])[0]

    def mset(self, keyvals):
        """
        Takes in a tuple of key/vals, example

        mset( [key1, 0.24], [key2, 0.60], ... )
        """

        kv2 = {}

        for key, val in keyvals:
            key = self.encode(key) if isinstance(key, dict) else key
            kv2[key] = float(val)

        super().mset(kv2)

    def df_get(self, keys):

        dx = pd.DataFrame(keys)
        n_labels = len(dx.label.unique())
        n_sequence = len(dx.sequence.unique())

        data = np.reshape(self.mget(keys), (n_sequence, n_labels))

        df = pd.DataFrame(
            data=data, columns=dx.label.unique(), index=dx.sequence.unique(),
        )

        return df

    def df_set(self, keys, df):
        x = [(a, b) for a, b in zip(keys, df.values.ravel())]
        self.mset(x)


if __name__ == "__main__":

    r = ZS_Redis_KeyStore()
    r.flushdb()

    key = {
        "hypothesis": "I like to go {}",
        "label": "shopping",
        "sequence": "I have money and I want to spend it.",
    }

    # Set a single hypothesis / response
    r.mset([(key, 0.245)])

    params = {
        "hypothesis_template": "I like to go {}",
        "candidate_labels": ["shopping", "swimming"],
        "sequences": [
            "I have a new swimsuit.",
            "Today is a new day for everyone",
            "I have money and I want to spend it.",
        ],
    }
    exit()

    keys = r.encode_series(
        params["hypothesis_template"], params["candidate_labels"], params["sequences"],
    )

    df = r.df_get(keys)
    print(df)

    #     shopping swimming
    #  0      None     None
    #  1      None     None
    #  2     0.245     None

    # Set some random data
    for col in df.columns:
        df[col] = np.random.uniform(size=len(df[col]))

    print(df)
    r.df_set(keys, df)

    print(r.df_get(keys))

    # Random data!
    #                                       shopping  swimming
    # I have a new swimsuit.                0.678845  0.224764
    # Today is a new day for everyone       0.664956  0.710711
    # I have money and I want to spend it.  0.599135  0.336866
