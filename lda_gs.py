import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

def main():
    X = np.load("preprocessing/data/train_prepared.npy").item()
    X_test = np.load("preprocessing/data/test_prepared.npy").item()

    lda = LatentDirichletAllocation(verbose=1,n_components=30)
    #search_params = {'n_components': [30]}
    #grid_search = GridSearchCV(lda, param_grid=search_params, cv=5, verbose=1, n_jobs=-1)
    lda.fit(X)

    joblib.dump(lda, "ldaxx.joblib")

    Y_test = lda.transform(X_test)

    print("training done .. computing silhouette score")
    pred = np.argmax(Y_test, axis=1)
    print(pred.shape)
    print(silhouette_score(X_test, pred, metric="cosine"))

    np.save("lda_result_xx", pred)

if __name__ == "__main__":
    main()