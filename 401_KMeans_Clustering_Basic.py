
##############################################################################
# K-MEANS CLUSTERING - BASIC TEMPELATE
##############################################################################

# IMPORT REQUIRED PYTHON PACKAGES

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

#IMPORT SAMPLE DATA

my_df = pd.read_csv("data/sample_data_clustering.csv")

# PLOT THE DATA

plt.scatter(my_df["var1"], my_df["var2"])
plt.xlabel("var1")
plt.ylabel("var2")
plt.show()

# INSTANTIATE & FIT THE MODEL

# The number of k can be set using n_clusters
kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(my_df)

# ADD THE CLUSTER LABELS TO OUR DF

my_df["cluster"] = kmeans.labels_
my_df["cluster"].value_counts()
my_df["cluster"].value_counts(normalize = True)

# PLOT OUR CLUSTERS AND CENTROIDS

centroids = kmeans.cluster_centers_
print(centroids)

clusters = my_df.groupby("cluster")

for cluster, data in clusters:
    plt.scatter(data["var1"], data["var2"], marker = "o", label = cluster)
    plt.scatter(centroids[cluster, 0], centroids[cluster, 1], marker = "X", color = "black", s = 300)
plt.legend()
plt.tight_layout()
plt.show()