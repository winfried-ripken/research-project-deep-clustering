import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
from tensorflow.python.keras import Input, Model
import numpy as np
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D
import os

nn_representations = 480
batch_size = 128
max_i = 30000

def get_model(wv_size, code_size):
    input = Input(shape=(wv_size,1))

    h = Conv1D(12, activation="relu", kernel_size=3)(input)
    h = Conv1D(8, activation="relu", kernel_size=3)(h)

    h = Dense(nn_representations, activation="relu")(Flatten()(h))
    output = Dense(code_size, activation="sigmoid")(h)

    model = Model(inputs=[input], outputs=[output])
    model.compile(optimizer="adam", loss="mse")

    return model, h, input

def test_predict(data, wv_size, code_size):
    sess = tf.Session()

    with sess.as_default():
        model, h, input = get_model(wv_size, code_size)
        model.load_weights("exp_binary_code.h5")

        nm = Model(inputs=[input], outputs=[h])
        result = nm.predict(data)

    return result

def test_entire_model():
    model = joblib.load("2stepclustering.joblib")
    test_w2v = np.load("test_dynamic.npy")
    binary_vector = np.load("binary_code.npy")

    representation_test = test_predict(test_w2v.reshape((test_w2v.shape[0], test_w2v.shape[1], 1)),
                                       test_w2v.shape[1], binary_vector.shape[1])

    return model.predict(representation_test)

def prepare_silhouette_test():
    test_w2v = np.load("test_dynamic.npy")
    train_w2v = np.load("train_dynamic.npy")
    binary_vector = np.load("binary_code.npy")

    print(train_w2v.shape)
    print(binary_vector.shape)

    representation_train = test_predict(train_w2v.reshape((train_w2v.shape[0], train_w2v.shape[1], 1)),
                                        train_w2v.shape[1], binary_vector.shape[1])
    representation_test = test_predict(test_w2v.reshape((test_w2v.shape[0], test_w2v.shape[1], 1)),
                                       train_w2v.shape[1], binary_vector.shape[1])

    return (representation_train, representation_test)

def test_silhouette(n_clusters, point_of_truth):
    (representation_train, representation_test) = prepare_silhouette_test()

    print("clustering")

    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(representation_train)
    cluster_labels = clusterer.predict(representation_test)
    joblib.dump(clusterer, "2stepclustering.joblib")

    print("silhouette score")

    silhouette_avg = silhouette_score(point_of_truth, cluster_labels, metric="cosine")
    print(silhouette_avg)

    return silhouette_avg

def km_clustering(nc, representation_train, representation_test, i):
    clusterer = KMeans(n_clusters=nc)
    clusterer.fit(representation_train)
    cluster_labels = clusterer.predict(representation_test)
    joblib.dump(clusterer, str(i) + "_2stepclustering.joblib")

    np.save(str(i) + "result", cluster_labels)
    return cluster_labels

def lda_clustering(nc, representation_train, representation_test, i):
    clusterer = LatentDirichletAllocation(n_components=nc,verbose=1)
    clusterer.fit(representation_train)
    Y_test = clusterer.transform(representation_test)
    joblib.dump(clusterer, str(i) + "_LDA_2stepclustering.joblib")

    cluster_labels = np.argmax(Y_test, axis=1)
    np.save(str(i) + "lda_result", cluster_labels)
    return cluster_labels

def grid_search_silhouette(point_of_truth, id, progress, register, clustering):
    cwd = os.getcwd()
    (representation_train, representation_test) = prepare_silhouette_test()

    n_clusters = [30] * 8

    register(id, ["Dir: " + str(cwd),
                  "Train shape: " + str(representation_train.shape),
                  "Test shape: " + str(representation_test.shape),
                  "Number of clusters to try: " + str(n_clusters)])

    best_cluster = 0
    best_sil = -1
    avg_sil = 0.0
    i = 0

    for nc in n_clusters:
        i += 1

        cluster_labels = clustering(nc, representation_train, representation_test, i)
        silhouette_avg = silhouette_score(point_of_truth, cluster_labels, metric="cosine")
        avg_sil += silhouette_avg

        if silhouette_avg > best_sil:
            best_sil = silhouette_avg
            best_cluster = i

        print("Cluster with Index=" + str(i) + " has score: " + str(silhouette_avg))
        progress(id, (int)((i * 100.0) / len(n_clusters)), ["Index: "+str(i), "Score: "+str(silhouette_avg)])

    #best_model = joblib.load(str(best_cluster) + "_2stepclustering.joblib")
    #joblib.dump(best_model, "2stepclustering.joblib")

    progress(id, 100, ["Best Index: "+str(best_cluster), "Best Score: "+str(best_sil), "Avg Score: "+str(avg_sil / len(n_clusters))])
    print("Best cluster with Index=" + str(best_cluster) + " has score: " + str(best_sil))

def test_silhouette_lda(n_clusters, point_of_truth):
    (representation_train, representation_test) = prepare_silhouette_test()

    print("clustering")

    clusterer = LatentDirichletAllocation(n_components=n_clusters)
    clusterer.fit(representation_train)
    Y_test = clusterer.transform(representation_test)

    print("silhouette score")

    pred = np.argmax(Y_test, axis=1)
    silhouette_avg = silhouette_score(point_of_truth, pred, metric="cosine")
    print(silhouette_avg)

    return silhouette_avg

def train(id, progress, register):
    sess = tf.Session()

    with sess.as_default():
        train = np.load("train_dynamic.npy")
        binary_vector = np.load("binary_code.npy")

        print(train.shape)
        print(binary_vector.shape)

        cwd = os.getcwd()

        register(id, ["Dir: " + str(cwd),
                      "Train shape: " + str(train.shape),
                      "Target shape: " + str(binary_vector.shape),
                      "Batch size: " + str(batch_size),
                      "Max i: " + str(max_i)])

        model, _, _ = get_model(train.shape[1], binary_vector.shape[1])
        loss = 0

        for i in range(1, max_i):
            # draw a sample
            indices = np.random.choice(binary_vector.shape[0], batch_size, replace=False)
            loss += model.train_on_batch(np.reshape(train[indices], (batch_size, train.shape[1], 1)), binary_vector[indices])

            if i % 1000 == 0:
                avg_loss = str(loss / 1000)
                print("avg loss: " + avg_loss)
                loss = 0

                print("making checkpoint at iterations: " + str(i))
                model.save_weights("exp_binary_code.h5")

                progress(id, int((i * 100.0) / max_i), ["Iterations: " + str(i),
                              "Avg loss: " + avg_loss])

def main(_):
    train("0", lambda *args: None, lambda *args: None)  # no logging

if __name__ == "__main__":
    tf.app.run()