from sre_constants import NOT_LITERAL_UNI_IGNORE
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)


# Erstelle K-Means
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_clustering = kmeans.predict(X)
# print(f"K-Means labels Attribute: \n{kmeans.labels_}")
# print(f"K-Means Cluster Center Attribute: \n{kmeans.cluster_centers_}")

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, marker="p")
plt.title("Datasets Truth (make blobs)")


plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=y_clustering, marker="p")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], marker="d", linewidths=2, c='#ff0000')
plt.title("Ground K-Means")

# Add Axes Labels
plt.xlabel("x1 axis")
plt.ylabel("x2 axis")

plt.show()
