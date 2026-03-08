import faiss
import numpy as np

class VectorDB:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, vectors, docs):

        self.index.add(np.array(vectors).astype('float32'))
        self.documents.extend(docs)

    def search(self, query_vector, k=5):

        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), k
        )

        results = []

        for idx in indices[0]:
            results.append(self.documents[idx])

        return results