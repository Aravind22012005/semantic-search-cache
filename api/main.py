from fastapi import FastAPI
from pydantic import BaseModel

from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder
from vector_store.vectordb import VectorDB
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache


app = FastAPI()


class Query(BaseModel):
    query: str


docs = load_dataset()

embedder = Embedder()

doc_embeddings = embedder.embed_documents(docs)

vectordb = VectorDB(len(doc_embeddings[0]))
vectordb.add(doc_embeddings, docs)

cluster_model = FuzzyCluster(n_clusters=10)
cluster_model.fit(doc_embeddings)

cache = SemanticCache(threshold=0.9)


@app.post("/query")
def query(q: Query):

    query_vector = embedder.embed_query(q.query)

    cache_result = cache.lookup(query_vector)

    if cache_result["hit"]:

        return {
            "query": q.query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity"],
            "result": cache_result["result"],
            "dominant_cluster": cache_result["cluster"]
        }

    results = vectordb.search(query_vector, k=1)

    cluster = cluster_model.dominant_cluster(query_vector)

    cache.store(
        q.query,
        query_vector,
        results[0],
        cluster
    )

    return {
        "query": q.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results[0],
        "dominant_cluster": cluster
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}