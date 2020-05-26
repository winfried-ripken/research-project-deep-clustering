import pyLDAvis
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.prepareTokenizer import prepare_vectorizer


def plot_grid_search_one(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    ax.plot(grid_param_1, scores_mean, '-o')

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()

def display_topics(model, X_test, test):
    # Create Document - Topic Matrix
    lda_output = model.transform(X_test)

    # column names
    topicnames = ["Topic" + str(i) for i in range(model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(X_test.shape[0])]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    df_document_topic['text'] = test

    # Styling
    def color_green(val):
        if isinstance(val, str):
            color = 'black'
        else:
            color = 'green' if val > .1 and val < 1 else 'black'

        return 'color: {col}'.format(col=color)

    def make_bold(val):
        if isinstance(val, str):
            weight = 400
        else:
            weight = 700 if val > .1 and val < 1 else 400

        return 'font-weight: {weight}'.format(weight=weight)

    # Apply Style
    df_document_topics = df_document_topic.head(150).style.applymap(color_green).applymap(make_bold)

    print(df_document_topics)

    writer = pd.ExcelWriter('output.xlsx')
    df_document_topics.to_excel(writer, 'Sheet1')
    writer.save()

def analyse_grid_search():
    model = joblib.load("grid_search.joblib")
    print(model.cv_results_)
    plot_grid_search(model.cv_results_, [10, 15, 20, 25, 50, 100], [.6, .9], "n_components", "learning_decay")

    train = np.load("preprocessing/data/train.npy")
    test = np.load("preprocessing/data/test.npy")

    tfidf_vectorizer, tokenize_func = prepare_vectorizer() # check this
    X = tfidf_vectorizer.fit_transform(train)
    X_test = tfidf_vectorizer.transform(test)

    display_topics(model.best_estimator_, X_test, test)

    viz = pyLDAvis.sklearn.prepare(model.best_estimator_, X, tfidf_vectorizer, n_jobs=1)
    # pyLDAvis.show(viz, open_browser=False)
    pyLDAvis.save_html(viz, 'lda.html')

def main():
    analyse_grid_search()

if __name__ == "__main__":
    main()