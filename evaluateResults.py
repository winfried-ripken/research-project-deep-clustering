import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.text import TSNEVisualizer
import os

from learn_predictor_for_binary_code import test_entire_model, test_silhouette

def vis_shilouette(X, cluster_labels):
    n_clusters = len(set(cluster_labels))

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels, metric="cosine")
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric="cosine")

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #
    # # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    #
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')

    # # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')
    #
    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')

    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

def analyse_results():
    rerun = False
    if ("rerun" in sys.argv):
        print("Redo everything")
        rerun = True

    X_test = np.load("test_prepared.npy").item()

    results = []
    names = []

    for filename in os.listdir("results"):
        if filename.endswith(".npy"):
            if filename[:-4] + "tsne.png" in os.listdir("results") and not rerun:
                continue

            results.append(np.load("results/"+filename))
            names.append(filename[:-4])

    for i in range(len(results)):
        print("iteration " + str(i+1) + " of " + str(len(results)) + " : " + names[i])

        vis_shilouette(X_test, results[i])
        plt.savefig("results/"+names[i]+"silhouette.png")

        plt.close()
        plt.figure()

        tsne = TSNEVisualizer(colormap=cm.get_cmap('jet', len(set(results[i][0:5000]))),
                              alpha=0.5, random_state=45) # make it deterministic
        tsne.fit(X_test[0:5000], ["c{}".format(c) for c in results[i][0:5000]])
        tsne.poof(outpath="results/"+names[i]+"tsne.png", clear_figure=True)

def analyse_2_step_model():
    X_test = np.load("test_prepared.npy").item()  # this is our Single point of truth
    #test_silhouette(30, X_test)

    test = X_test[0:1000]
    prediction = test_entire_model()[0:1000]

    vis_shilouette(test, prediction)
    plt.savefig("silhouette.png")

    tsne = TSNEVisualizer(colormap=cm.get_cmap('jet', len(set(prediction))))
    tsne.fit(test[0:1000], ["c{}".format(c) for c in prediction])
    tsne.poof(outpath="tsne.png")

def main():
    analyse_results()

if __name__ == "__main__":
    main()