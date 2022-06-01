
##############################################################################
# K-MEANS CLUSTERING - ADVANCED TEMPELATE
##############################################################################

# IMPORT REQUIRED PYTHON PACKAGES

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE THE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# IMPORT TABLES

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name = "product_areas")

# MERGE ON PRODUCT AREA NAME

transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")

# DROP THE NON-FOOD CATEGORY

transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# AGGREGATE SALES AT CUSTOMER LEVEL (BY PRODUCT AREA)

transaction__summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# PIVOT DATA TO PLACE PRODUCT AREAS AS COLUMNS

# To avoid that we don't end up with both customer_id and product area name in the index.
transaction__summary_pivot = transactions.pivot_table(index = "customer_id",
                                                      columns = "product_area_name",
                                                      values = "sales_cost",
                                                      aggfunc = "sum",
                                                      fill_value = 0,
                                                      margins = True,
                                                      margins_name = "Total").rename_axis(None, axis = 1)

# TURN SALES INTO % SALES

# Axis = 0, ensures we do it a row level
transaction__summary_pivot = transaction__summary_pivot.div(transaction__summary_pivot["Total"], axis = 0)

# DROP THE "TOTAL" COLUMN

data_for_clustering = transaction__summary_pivot.drop(["Total"], axis = 1)

# ~~~~~~~~~~~~~~~~~~~~~~~ DATA PREPARATION AND CLEANING ~~~~~~~~~~~~~~~~~~~~~~

# CHECK FOR MISSING VALUES

data_for_clustering.isna().sum()

# NORMALIZE DATA

# Even though our data is is percentage and hence between 0 and 1.
# It is sometimes better to do min-max scaling to ensure it is not biased towards a few values.
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)

# ~~~~~~~~~~~~~~~~~~~ USE WCSS TO FIND A GOOD VALUE FOR K ~~~~~~~~~~~~~~~~~~~~

k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values, wcss_list)
plt.title("Within Cluster Sum of Squares - by K")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~ INSTANTIATE & FIT MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(data_for_clustering_scaled)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ USE CLLUSTER INFORMATION ~~~~~~~~~~~~~~~~~~~~~~~~~

# ADD CLUSTER LABELS TO OUR DATA

data_for_clustering["cluster"] = kmeans.labels_

# CHECK CLUSTER SIZES

data_for_clustering["cluster"].value_counts()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFILE OUR CLUSTERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~

cluster_summary = data_for_clustering.groupby("cluster")[["Dairy", "Fruit", "Meat", "Vegetables"]].mean()
