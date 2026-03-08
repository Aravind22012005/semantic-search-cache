import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.9):

        self.cache = []
        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_vector):

        for entry in self.cache:

            sim = cosine_similarity(
                [query_vector],
                [entry["vector"]]
            )[0][0]

            if sim >= self.threshold:

                self.hit_count += 1

                return {
                    "hit": True,
                    "result": entry["result"],
                    "matched_query": entry["query"],
                    "similarity": float(sim),
                    "cluster": entry["cluster"]
                }

        self.miss_count += 1

        return {"hit": False}

    def store(self, query, vector, result, cluster):

        self.cache.append({
            "query": query,
            "vector": vector,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total > 0 else 0
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0