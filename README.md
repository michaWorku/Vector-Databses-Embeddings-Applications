# Vector Databases: From Embeddings to Applications  

## Course Overview  
[Vector Databases: From Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) explores how **vector databases** enhance AI applications like **semantic search, image recognition, recommender systems, and Retrieval-Augmented Generation (RAG)**. This course provides hands-on experience in building efficient applications with **vector search, hybrid search, and multilingual search**.  

### **What You'll Learn**  
- **Understand vector databases** and their role in AI-powered applications.  
- **Build embeddings** from images and text for efficient similarity search.  
- **Explore search techniques** including **k-nearest neighbors (KNN), approximate nearest neighbors (ANN), and Hierarchical Navigable Small World (HNSW) algorithms**.  
- **Develop AI applications** using **RAG, hybrid search, and multilingual search**.  
- **Optimize search performance** with efficient **indexing and retrieval methods**.  

By the end of this course, you’ll be equipped to **integrate vector databases into real-world AI applications**, improving search accuracy, personalization, and retrieval efficiency. 🚀  

## Course Contents  

### [**1. Introduction to Vector Databases**]()  
- **What are vector databases?**  
  - Applications in **NLP, image recognition, recommender systems, and semantic search**.  
  - Role in **LLM applications and Retrieval-Augmented Generation (RAG)**.  
- **Understanding embeddings**  
  - How vector embeddings **capture meaning**.  
  - Representing text and images in a machine-readable format.  

### [**2. Obtaining Vector Representations of Data**]()  
- **Creating Embeddings**  
  - **Image embeddings** using **Variational Autoencoders (VAE)**.  
  - **Text embeddings** with **Transformer-based models**.  
- **Measuring similarity between embeddings**  
  - **Euclidean Distance (L2), Manhattan Distance (L1), Dot Product, and Cosine Similarity**.  

### [**3. Searching for Similar Vectors**]()  
- **K-Nearest Neighbors (KNN) Search**  
  - **Brute-force search algorithm**: Compute distances between query and stored vectors.  
  - **Runtime complexity: O(dN)**.  
- **Approximate Nearest Neighbors (ANN)**  
  - **Trade-off between speed and accuracy**.  
  - **Hierarchical Navigable Small World (HNSW) algorithm**: **O(log(N)) search time complexity**.  
  - **Efficient indexing for large-scale vector searches**.  

### [**4. Vector Databases in Practice**]()  
- **CRUD operations** (Create, Read, Update, Delete) in vector databases.  
- **Working with Weaviate**  
  - **Importing data** and generating embeddings.  
  - **Running vector searches** using embeddings.  
  - **Filtering results** based on similarity scores.  

### [**5. Sparse, Dense, and Hybrid Search**]()  
- **Dense Search (Semantic Search)**  
  - Uses **vector embeddings** to find semantically similar data.  
  - **Challenges:** Poor performance on **out-of-domain data, serial numbers, and keyword-based queries**.  
- **Sparse Search (Keyword Search)**  
  - Uses **Bag-of-Words (BoW)** and **BM25** for exact keyword matching.  
- **Hybrid Search**  
  - **Combining sparse and dense search** for better accuracy.  
  - Uses **scoring systems** to rank results effectively.  

### [**6. Building AI Applications with Vector Databases**]()  
- **Multilingual Search**  
  - Embeddings enable **cross-language retrieval**.  
- **Retrieval-Augmented Generation (RAG)**  
  - **Using vector databases to enhance LLM responses**.  
  - **Advantages:**  
    - **Reduces hallucinations** in LLM-generated responses.  
    - **Enables citation of sources** for transparency.  
    - **Improves response accuracy for knowledge-intensive tasks**.  
- **The RAG Workflow**  
  1. Query a **vector database**.  
  2. Retrieve **relevant documents**.  
  3. **Incorporate retrieved data** into the prompt.  
  4. Generate a **context-aware response using an LLM**.  

## Notebooks  
1. **[Creating Embeddings]()** – Generate vector embeddings for images and text.  
2. **[KNN]()** – Implement KNN. 
3. **[ANN]()** – Implement ANN. 
4. **[Vector Databases]()** – Store and retrieve embeddings with Weaviate.  
5. **[Sparse, Dense & Hybrid Search]()** – Compare different search methods.  
6. **[Building AI Applications]()** – Implement **multilingual search & RAG**.  

## Getting Started  
1. Install dependencies and set up your environment.  
2. Load data and generate embeddings.  
3. Implement **vector search, hybrid search, and RAG** applications.  

## References  
- [Course Link](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)  

This course provides the foundational skills to **build AI-powered applications with vector databases**, enabling faster, more accurate, and **context-aware search and retrieval**. 🚀