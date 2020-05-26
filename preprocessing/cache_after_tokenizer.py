import numpy as np

from preprocessing.prepareTokenizer import prepare_vectorizer


def main():
    train = np.load("data/train.npy")
    test = np.load("data/test.npy")

    print("Datasets loaded. Starting preprocessing ..")

    tfidf_vectorizer, tokenize_func = prepare_vectorizer(3, 5000)
    X = tfidf_vectorizer.fit_transform(train)
    X_test = tfidf_vectorizer.transform(test)

    np.save("data/train_prepared_5000_feat.npy", X)
    np.save("data/test_prepared_5000_feat.npy", X_test)

if __name__ == "__main__":
    main()