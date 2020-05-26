import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer


def main():
    X = np.load("preprocessing/data/train_prepared.npy").item()
    lda = LatentDirichletAllocation(verbose=1, max_iter=1, n_components=10)
    Y = lda.fit_transform(X)
    print(lda.components_.shape)
    #X_test = np.load("test_prepared.npy").item()
    #Y_test = lda.transform(X_test)
    #pred = np.argmax(Y_test, axis=1)

    #print(silhouette_score(X_test, pred, metric="cosine"))

    one_hot = np.zeros_like(Y)
    one_hot[np.arange(len(Y)), Y.argmax(1)] = 1
    #print(one_hot.shape)

    #binary_vector = np.asarray(LabelBinarizer().fit_transform(one_hot))

    np.save("binary_code", Y)

if __name__ == "__main__":
    main()