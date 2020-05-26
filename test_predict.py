from sklearn.externals import joblib
import numpy as np
import os
import pandas as pd

from learn_predictor_for_binary_code import test_entire_model

def save_test_predict_lda():
    model = joblib.load("grid_search.joblib")
    print(model.best_params_)
    X_test = np.load("preprocessing/data/test_prepared.npy").item()

    Y_test = model.best_estimator_.transform(X_test)
    pred = np.argmax(Y_test, axis=1)

    np.save("lda_result", pred)

def save_test_predict_2_step_model(name="2_step_result"):
    prediction = test_entire_model()
    np.save(name, prediction)

def aggregate_results():
    test = np.load("preprocessing/data/test.npy")

    results = []
    names = []

    for filename in os.listdir("results"):
        if filename.endswith(".npy"):
            results.append(np.load("results/"+filename))
            names.append(filename[:-4])

    docnames = ["Doc" + str(i) for i in range(test.shape[0])]
    df_document_topic = pd.DataFrame(index=docnames)

    df_document_topic['text'] = test

    for r in range(len(results)):
        df_document_topic[names[r]] = results[r]

    writer = pd.ExcelWriter('aggregated_results.xlsx')
    df_document_topic.to_excel(writer, 'Sheet1')
    writer.save()

def main():
    aggregate_results()

if __name__ == "__main__":
    main()