import numpy as np

class KMeans:
    def __init__(self, k):

        self.k = k
        self.centroids = None

    def fit(self, data):

        n_samples = data.shape[0]

        # Randomly initialize k centroids
        self.centroids = data[np.random.choice(n_samples, self.k, replace=False)]

        while (True) :
            # Assign clusters
            distances = np.array([[np.sqrt(np.sum((point - centroid) ** 2)) for centroid in self.centroids] for point in data])
            cluster_labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.array_equal(new_centroids, self.centroids):
                break

            self.centroids = new_centroids


    def predict(self, data):

        distances = np.array([[np.sqrt(np.sum((point - centroid) ** 2)) for centroid in self.centroids] for point in data])
        cluster_labels = np.argmin(distances, axis=1)
        
        return cluster_labels


