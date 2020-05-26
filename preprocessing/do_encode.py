from encoder import SHEncoder
import numpy as np
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
X = np.load("train_dynamic.npy")
print(X.shape)

encoder = SHEncoder()
#encoder.build({"vals":X,"nbits":32})
#save_obj(encoder.ecdat, "ecdat")
encoder.ecdat = load_obj("ecdat")
#print(load_obj("ecdat"))

encoded = encoder.encode(X)[1]
#np.save("binary_code", encoded)

print(len(np.unique(encoder.encode(X[0:100000])[0], axis=0)))
print(X.shape)
#X=X.todense()
print(type(X))
print(len(np.unique(X[0:100000], axis=0)))