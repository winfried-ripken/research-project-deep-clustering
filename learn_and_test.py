import time

import sys

from learn_predictor_for_binary_code import test_silhouette, train, test_silhouette_lda, grid_search_silhouette, \
    lda_clustering

import os
import numpy as np
import requests
import tensorflow as tf

from test_predict import save_test_predict_2_step_model


def register(id, details):
    d_string = ""

    for d in details:
        d_string += str(d) + "</br>"

    requests.post('http://127.0.0.1:5000/register', json={'Id': id, 'Details': d_string[:-5]})

def deregister(id):
    requests.post('http://127.0.0.1:5000/deregister', json={'Id': id})

def progress(id, n, status):
    s_string = ""

    for s in status:
        s_string += str(s) + "</br>"

    requests.post('http://127.0.0.1:5000/status', json={'Id': id, 'Progress': str(n), 'Status': s_string[:-5]})


def exp_train_neural_network():
    id = str(time.time())  # id is starting time
    train(id, progress, register)

    X_test = np.load("test_prepared.npy").item()  # this is our Single point of truth

    avg_score = 0.0
    avg_i = 3
    for i in range(avg_i):
        score = test_silhouette(30, X_test)
        avg_score += score
        progress(id, 100, ["Silhouette Score " + str(i + 1) + " for 30 clusters (KMeans): " + str(score)])

    progress(id, 100, ["Average Silhouette Score for 30 clusters (KMeans): " + str(avg_score / avg_i)])

    # score = test_silhouette_lda(5, X_test)
    # progress(id, 100, ["Silhouette Score for 5 clusters (LDA): " + str(score)])

    deregister(id)

def exp_grid_search_silhouette():
    id = str(time.time())  # id is starting time

    X_test = np.load("test_prepared.npy").item()
    grid_search_silhouette(X_test, id, progress, register, lda_clustering)

    #dirpath = os.getcwd()
    #foldername = os.path.basename(dirpath)

    #save_test_predict_2_step_model(foldername)
    deregister(id)

def main(_):
    if len(sys.argv) < 2:
        print("Missing argument: 1 for NN training, 2 for silhouette grid search")
        exit()

    if str(sys.argv[1]) == "1":
        exp_train_neural_network()
    elif str(sys.argv[1]) == "2":
        exp_grid_search_silhouette()
    else:
        print("Invalid argument: 1 for NN training, 2 for silhouette grid search")
        exit()

if __name__ == "__main__":
    if ("gpu" in sys.argv):
        print("Training on GPU")
    else:
        print("Training on CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    tf.app.run()