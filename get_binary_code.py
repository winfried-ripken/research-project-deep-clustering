import joblib
import numpy as np
import time

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from encoder import SHEncoder


def main():
    X = np.load("preprocessing/data/train_prepared_small.npy").item()
    print(X.shape)
    # create Product Quantization Indexer
    encoder = SHEncoder()
    encoder.build({"vals":X,"nbits":48})
    # set storage for the indexer, the default is store indexex in memory
    #idx.set_storage()
    # build indexer
    #idx.build({'vals': X_db, 'nsubq': 8})
    # add database items to the indexer
    #idx.add(X_db)
    # searching in the database, and return top-100 items for each query
    #idx.search(X_qry, 100)

    # cos_sim = csr_matrix((X.shape[0],X.shape[0]))
    #
    # print(X.shape[0],X.shape[0])
    #
    #
    # print("calculating cosine similarities")
    #
    # start = time.time()
    # cos_sim[1,2] = cosine_similarity(X[1], X[2])
    # cos_sim[1,12] = cosine_similarity(X[1], X[12])
    # cos_sim[1,22] = cosine_similarity(X[1], X[22])
    # cos_sim[1,32] = cosine_similarity(X[1], X[32])
    # cos_sim[1,42] = cosine_similarity(X[1], X[42])
    # cos_sim[1,52] = cosine_similarity(X[1], X[52])
    # cos_sim[1,62] = cosine_similarity(X[1], X[62])
    # cos_sim[1,72] = cosine_similarity(X[1], X[72])
    # cos_sim[1,82] = cosine_similarity(X[1], X[82])
    # cos_sim[1,92] = cosine_similarity(X[1], X[92])
    # end = time.time()
    # print(X.shape[0])
    # print((end - start))
    # print(((end - start) * X.shape[0] * X.shape[0]) / (10 * 60 * 60 * 24 * 365))
    #
    # start = time.time()
    # cos_sim[1, 2] = (X[1] * X[2]) #/
    # cos_sim[1, 12] = cosine_similarity(X[1], X[12])
    # cos_sim[1, 22] = cosine_similarity(X[1], X[22])
    # cos_sim[1, 32] = cosine_similarity(X[1], X[32])
    # cos_sim[1, 42] = cosine_similarity(X[1], X[42])
    # cos_sim[1, 52] = cosine_similarity(X[1], X[52])
    # cos_sim[1, 62] = cosine_similarity(X[1], X[62])
    # cos_sim[1, 72] = cosine_similarity(X[1], X[72])
    # cos_sim[1, 82] = cosine_similarity(X[1], X[82])
    # cos_sim[1, 92] = cosine_similarity(X[1], X[92])
    # end = time.time()
    # print((end - start))
    #
    # exit()
    # for i in range(X.shape[0]):
    #     for j in range(i, X.shape[0]):
    #         cos_sim[i,j] = cosine_similarity(X[i], X[j])

if __name__ == "__main__":
    main()