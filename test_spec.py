import numpy as np
from sklearn.manifold import SpectralEmbedding

train = np.load("preprocessing/data/train_prepared.npy")
embedding = SpectralEmbedding(n_components=10)
print(embedding.fit_transform(train))