import numpy as np
import skfuzzy as fuzz


class FuzzyCluster:

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.centers = None
        self.membership = None

    def fit(self, embeddings):

        data = np.array(embeddings).T

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data,
            c=self.n_clusters,
            m=2,
            error=0.005,
            maxiter=1000
        )

        self.centers = cntr
        self.membership = u

        return u

    def dominant_cluster(self, vector):

        vector = np.array(vector).reshape(-1, 1)

        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            vector,
            self.centers,
            m=2,
            error=0.005,
            maxiter=1000
        )

        return int(np.argmax(u))