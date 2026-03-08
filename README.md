# semantic-search-cache
# Semantic Search Cache System

## Overview

This project implements a **semantic caching system for natural language queries**.
Instead of recomputing responses for similar queries, the system stores previous query embeddings and retrieves cached results when semantically similar queries are detected.

This significantly **reduces response time and computation cost** for repeated or similar queries.

The system uses **vector embeddings, cosine similarity, and clustering techniques** to determine cache hits.

---

# System Architecture

Query → Embedding Generator → Vector Database
→ Similarity Search → Cache Hit / Miss
→ Response Retrieval or Generation

Components work together to determine whether a new query is similar enough to an existing query in the cache.

---

# Project Structure

```
semantic-search-cache/
│
├── api/
│   └── main.py                # FastAPI application and API endpoints
│
├── cache/
│   └── semantic_cache.py      # Core semantic caching logic
│
├── clustering/
│   └── fuzzy_cluster.py       # Fuzzy clustering for grouping similar queries
│
├── data/
│   └── dataset_loader.py      # Loads and processes dataset
│
├── embeddings/
│   └── embedder.py            # Generates sentence embeddings
│
├── vector_store/
│   └── vectordb.py            # Vector database operations using FAISS
│
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── .dockerignore
└── README.md
```

---

# Key Features

* Semantic similarity detection using **Sentence Transformers**
* Fast vector search with **FAISS**
* Cache hit detection using **cosine similarity**
* Query clustering using **fuzzy clustering**
* REST API built with **FastAPI**
* Fully **Dockerized for easy deployment**

---

# Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS (Facebook AI Similarity Search)
* NumPy
* Scikit-learn
* Docker

---

# API Endpoint

### Query Endpoint

```
POST /query
```

### Request Body

```json
{
  "query": "computer graphics rendering"
}
```

### Example Response

```json
{
  "query": "computer graphics rendering",
  "cache_hit": true,
  "matched_query": "computer graphics rendering",
  "similarity_score": 1.0,
  "result": "cached response",
  "dominant_cluster": 1
}
```

---

# Running the Project Locally

## 1 Install Dependencies

```
pip install -r requirements.txt
```

## 2 Run API Server

```
uvicorn api.main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Swagger documentation is available at:

```
http://127.0.0.1:8000/docs
```

---

# Running with Docker

## Build the Docker Image

```
docker build -t semantic-search-api .
```

## Run the Container

```
docker run -p 8000:8000 semantic-search-api
```

The API will be available at:

```
http://localhost:8000
```

---

# How the Semantic Cache Works

1. A user query is received via the API.
2. The query is converted into a vector embedding.
3. The embedding is compared against cached embeddings using cosine similarity.
4. If similarity exceeds a defined threshold:

   * Cache hit occurs
   * Cached result is returned.
5. If not:

   * A new response is generated
   * Query and embedding are stored in cache.

---

# Future Improvements

* Persistent vector database storage
* Redis integration for distributed caching
* GPU acceleration for embedding generation
* Adaptive similarity thresholding
* Query analytics dashboard

---

# Author

Sai Aravind Sannidhanam
Computer Science (AI & Robotics)
Vellore Institute of Technology, Chennai
