import redis
import json

class ZS_Redis_KeyStore(redis.Redis):

    def encode(self, item):
        '''
        Expect item to have the following keys:
        hypothesis, label, sequence
        '''

        assert 'hypothesis' in item
        assert 'label' in item
        assert 'sequence' in item
        assert len(item.keys()) == 3

        return json.dumps(key)

    
    def mget(self, keys):
        '''
        Before multiget, encode the input parameters.
        '''
        
        keys = [
            self.encode(key) if isinstance(key, dict) else key
            for key in keys
        ]
       
        vals = super().mget(keys)

        vals = [x if x is None else float(x) for x in vals]
        return vals

    def get(self, key):
        return self.mget([key])[0]
        
    
    def mset(self, keyvals):
        '''
        Takes in a tuple of key/vals, example

        mset( [key1, 0.24], [key2, 0.60], ... )
        '''
        
        kv2 = { }

        for key, val in keyvals:
            key = self.encode(key) if isinstance(key, dict) else key
            kv2[key] = float(val)

        super().mset(kv2)
    

if __name__ == "__main__":
    
    r = ZS_Redis_KeyStore()
    r.flushdb()

    hypothesis_template = "I like to go {}"
    candidate_labels = ["shopping", "swimming"]
    sequence = 'I have money and I want to spend it.'
    
    key = {
        'hypothesis' : hypothesis_template,
        'label' : candidate_labels[0],
        'sequence' : sequence,
    }

    r.mset([(key, 0.245)])

    print(r.keys())
    print(r.get(key))
    print(r.mget([key, key]))
