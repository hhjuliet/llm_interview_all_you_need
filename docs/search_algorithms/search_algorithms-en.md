# What are architecture patterns for information retrieval & semantic search?

Information retrieval and semantic search systems employ several architectural patterns optimized for different requirements:

1. **Classic Retrieval Architecture**
   - Components: Tokenizer → Inverted Index → BM25 Scorer → Results
   - Best for: Keyword-heavy domains like legal document search
   - Limitations: Struggles with semantic similarity

2. **Pure Vector Search Architecture**
   - Components: Embedding Model → Vector Index → Similarity Search
   - Strengths: Handles semantic relationships and synonyms
   - Example: Finding conceptually similar research papers

3. **Hybrid Search Architecture**
   ```
   graph LR
     A[Query] --> B(Keyword Search)
     A --> C(Vector Search)
     B & C --> D{Reranker}
     D --> E[Final Results]
   ```
   - Combines precision of keyword search with recall of vector search
   - Fusion methods: Reciprocal Rank Fusion (RRF), weighted scoring

4. **Multi-Stage Retrieval**
   - Phases:
     1. Candidate Generation (vector/keyword → 1000 results)
     2. Reranking (cross-encoder → top 10)
     3. Generative Synthesis (LLM → final answer)
   - Used in: Google's search engine, Bing

5. **Federated Search Architecture**
   - Queries multiple specialized indexes (e.g., product DB + knowledge base)
   - Merging: Weighted fusion based on domain authority
   - Example: Enterprise search across departments

# Why it's important to have very good search

High-quality search systems deliver critical business value across domains:

1. **User Experience Impact**
   - 53% abandonment rate when search fails (Forrester)
   - 2-second delay → 4% conversion drop (Akamai)
   - Personalization increases engagement by 40%

2. **Operational Efficiency**
   - 30% reduction in support tickets via self-service
   - 45% faster information retrieval in enterprises
   - Prevents duplicate work through knowledge discovery

3. **Competitive Advantage**
   - 68% premium for superior search experience (Gartner)
   - Differentiates products (e.g., GitHub vs Bitbucket code search)
   - Enables data-driven decision making

4. **Risk Mitigation**
   - Critical in healthcare: Prevents missed diagnoses
   - Legal compliance: Ensures complete document retrieval
   - Financial: Avoids costly information gaps

5. **Monetization Potential**
   - E-commerce: 35% revenue from search-driven recommendations
   - Media: 50% longer session duration with relevant content

# How can you achieve efficient and accurate search results in large-scale datasets?

Implement this 7-point framework for billion-scale datasets:

1. **Tiered Indexing Strategy**
   ```
   hot_data = realtime_index(embeddings)        # HNSW in RAM
   warm_data = compressed_hnsw(embeddings)      # IVF_PQ on SSD
   cold_data = disk_ann(quantized_embeddings)   # ScaNN on cold storage
   ```

2. **Hybrid Retrieval Optimization**
   - Combine sparse (BM25) + dense (vector) + metadata filters
   - Fusion algorithm: RRF (Reciprocal Rank Fusion)
   - Dynamic weighting based on query type detection

3. **Hardware Acceleration**
   - GPU-optimized vector search (FAISS-GPU)
   - FPGA for custom distance computations
   - In-memory caching for frequent queries (Redis)

4. **Query Optimization Pipeline**
   - Query classification → routing to specialized indexes
   - Automatic spelling correction + synonym expansion
   - Dynamic pruning based on early results

5. **Proactive Performance Monitoring**
   ```
   metrics = {
     "latency_99th": <50ms,
     "recall@100": >0.95,
     "error_rate": <0.1%,
     "throughput": >1000 qps
   }
   ```

6. **Continuous Relevance Tuning**
   - A/B test ranking models weekly
   - Hard negative mining from query logs
   - Embedding model refresh every 3 months

7. **Scalability Patterns**
   - Sharding by tenant/region
   - Streaming index updates (Kafka → Flink)
   - Cold storage archiving for raw documents

# Consider a scenario where a client has already built a RAG-based system that is not giving accurate results, upon investigation you find out that the retrieval system is not accurate, what steps you will take to improve it?

Execute this 5-phase remediation protocol:

### Phase 1: Root Cause Analysis
1. Run diagnostic tests:
   ```
   recall@k = measure_retrieval_accuracy(test_queries)
   error_patterns = classify_errors(missed_results)
   ```
   Common failure modes:
   - Synonym mismatch (42%)
   - Temporal mismatch (28%)
   - Context fragmentation (18%)

### Phase 2: Immediate Interventions
1. Query Enhancement:
   ```
   expanded_query = llm_generate("Expand query with synonyms: {query}")
   ```
2. Hybrid Retrieval Fallback:
   Add BM25 as secondary retriever

### Phase 3: Medium-Term Improvements
1. Chunking Optimization:
   - Implement content-aware chunking
   - Add 15% token overlap
   - Atomic table/list preservation
2. Embedding Model Upgrade:
   ```
   model.fit(domain_data, loss=MultipleNegativesRankingLoss())
   ```

### Phase 4: Architectural Changes
1. Multi-Stage Retrieval:
   - Stage 1: Vector search → 100 candidates
   - Stage 2: Cross-encoder reranker → top 5
2. Metadata Enrichment:
   - Time-decay boosting for recent documents
   - Entity-aware relevance weighting

### Phase 5: Validation & Monitoring
1. Establish test suite with 200 edge-case queries
2. Implement continuous monitoring:
   ```
   alert_if recall@5 < 0.85 for 3 consecutive days
   ```

# Explain the keyword-based retrieval method

Keyword-based retrieval relies on lexical matching through these core mechanisms:

### Core Components
1. **Inverted Index Construction**
   ```
   Document 1: "The quick brown fox"
   Document 2: "Fox jumps over dog"
   
   Inverted Index:
   the → [1]
   quick → [1]
   brown → [1]
   fox → [1,2]
   jumps → [2]
   over → [2]
   dog → [2]
   ```

2. **Scoring Models**
   - **TF-IDF**:
     ```
     score = term_frequency * log(total_docs / docs_with_term)
     ```
   - **BM25 (Improved)**:
     ```
     score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length/avg_dl))
     ```

3. **Query Processing Pipeline**
   ```
   Raw Query → Tokenization → Stemming → Stopword Removal → 
   Index Lookup → Score Calculation → Ranking
   ```

### Optimization Techniques
| **Problem** | **Solution** | **Implementation** |
|-------------|--------------|-------------------|
| Synonym mismatch | Query expansion | WordNet, LLM synonym generation |
| Spelling errors | Fuzzy matching | Levenshtein distance, phonetic algorithms |
| Context blindness | Entity-aware boosting | Named entity recognition weighting |

### Performance Characteristics
- **Latency**: 1-10ms for million-document indexes
- **Accuracy**: 60-80% for exact match queries
- **Memory**: 20-30% of original text size

# How to fine-tune re-ranking models?

Fine-tune rerankers using this 5-step process:

### 1. Dataset Preparation
- Collect 10K+ query-document pairs
- Label relevance: 0 (irrelevant) to 4 (perfect)
- Include hard negatives (semantically close but irrelevant)
- Split: 70% train, 20% validation, 10% test

### 2. Model Selection
| **Model Type** | **Latency** | **Accuracy** | **Use Case** |
|----------------|-------------|--------------|-------------|
| Cross-Encoder (MiniLM-L6) | 4ms | 85% | Real-time systems |
| Cross-Encoder (Electra-base) | 15ms | 92% | High-accuracy |
| Dual-Encoder | 1ms | 78% | Pre-filtering |

### 3. Training Configuration
```
model.fit(
  train_data,
  loss=CosineSimilarityLoss(),
  optimizer=AdamW(lr=2e-5),
  epochs=3,
  batch_size=32,
  warmup_steps=500
)
```

### 4. Advanced Techniques
- **Knowledge Distillation**:
   ```
   teacher = large_cross_encoder
   student = small_dual_encoder
   loss = MSE(teacher_logits, student_logits)
   ```
- **Contrastive Fine-tuning**:
   ```
   Positive: relevant doc
   Negative: hard negative + random negative
   ```

### 5. Deployment Optimization
- ONNX conversion for 2x speedup
- Dynamic model selection:
   ```
   if query_complexity > threshold:
        use_large_model
   else:
        use_small_model
   ```
- A/B testing framework:
   ```
   ab_test(reranker_v1, reranker_v2, metric=NDCG@5)
   ```
# Explain most common metric used in information retrieval and when it fails?

The most common metric in information retrieval is **Mean Average Precision (MAP)**, which measures ranking quality across multiple queries. Here's a detailed explanation:

### MAP Calculation
1. For each query:
   ```
   Precision@k = (# relevant docs in top k) / k
   Average Precision (AP) = ∑(Precision@k × rel_k) / total_relevant
   ```
2. MAP = mean(AP across all queries)

### When MAP Fails
| **Failure Scenario** | **Reason** | **Alternative Metric** |
|----------------------|-----------|-----------------------|
| Graded Relevance (e.g., 1-5 stars) | Treats all relevant docs equally | nDCG (Normalized Discounted Cumulative Gain) |
| Position Sensitivity | Doesn't sufficiently penalize late relevant results | MRR (Mean Reciprocal Rank) |
| Variable Result Set Size | Unreliable when relevant docs < 5 | Recall@k |
| User Behavior Modeling | Doesn't model real user interaction | Expected Reciprocal Rank (ERR) |
| Binary Relevance Limitations | Can't handle partial relevance | Rank-Biased Precision (RBP) |

### Technical Limitations
1. **Assumption Violation**:
   ```
   Fails when relevance isn't binary (e.g., 3-star vs 5-star docs)
   ```
2. **Positional Blindness**:
   ```
   Doesn't distinguish between:
   Ranking A: [relevant, irrelevant, relevant]
   Ranking B: [irrelevant, relevant, relevant]
   Both have same AP despite different user experience
   ```

# If you were to create an algorithm for a Quora-like question-answering system, with the objective of ensuring users find the most pertinent answers as quickly as possible, which evaluation metric would you choose to assess the effectiveness of your system?

For a Quora-like Q&A system prioritizing quick access to pertinent answers, select **Mean Reciprocal Rank (MRR)**:

### Why MRR?
1. **Speed Focus**:
   ```
   MRR = average(1/rank_first_relevant_answer)
   ```
   Directly measures how quickly users find the first satisfactory answer

2. **User Behavior Alignment**:
   - 72% of users never scroll past first answer (Google study)
   - Perfectly matches "quick pertinent answer" objective

3. **Interpretability**:
   ```
   MRR = 0.8 → average position of first relevant answer = 1.25
   ```

### Implementation Example
```
queries = ["What is quantum computing?"]
results = {
  "q1": [answer3 (relevant), answer1, answer2]  # rank=1 → score=1
  "q2": [answer1, answer3 (relevant), answer2]  # rank=2 → score=0.5
}
MRR = (1 + 0.5)/2 = 0.75
```

### Complementary Metrics
1. **Time-to-First-Click**: Measure real user behavior
2. **Answer Satisfaction Rate**: Post-answer survey (1-5 scale)
3. **Abandonment Rate**: % queries with no click

# I have a recommendation system, which metric should I use to evaluate the system?

For recommendation systems, use **Normalized Discounted Cumulative Gain (nDCG)** as the primary metric:

### Why nDCG?
1. **Handles Graded Relevance**:
   ```
   DCG = ∑(rel_i / log2(i+1))
   nDCG = DCG / ideal_DCG
   ```
   Perfect for star ratings (e.g., 1-5 stars)

2. **Position Discounting**:
   Naturally weights top positions higher

3. **Normalization**:
   Allows comparison across different queries/users

### Implementation Example
```
# User's actual engagement: [5-star, 3-star, 1-star]
DCG = 5/log2(2) + 3/log2(3) + 1/log2(4) ≈ 5 + 1.89 + 0.5 = 7.39

# Ideal ordering: [5-star, 3-star, 1-star] → same DCG
nDCG = 7.39/7.39 = 1.0
```

### Industry-Specific Supplements
| **System Type** | **Additional Metrics** |
|-----------------|------------------------|
| E-commerce | Conversion Rate, Revenue per Visit |
| Content Streaming | Completion Rate, Skip Rate |
| Social Media | Share Rate, Dwell Time |

# Compare different information retrieval metrics and which one to use when?

### Metric Comparison Table
| **Metric** | **Formula** | **Strengths** | **Weaknesses** | **Best For** |
|------------|-------------|---------------|----------------|--------------|
| **Precision@k** | (# relevant in top k)/k | Simple, intuitive | Ignores ranking order | Known-item search |
| **Recall@k** | (# relevant found)/total_relevant | Measures coverage | Doesn't consider position | Legal discovery |
| **MAP** | Mean(Average Precision) | Comprehensive ranking quality | Binary relevance only | Academic benchmarks |
| **MRR** | Mean(1/rank_first_relevant) | Speed-focused | Ignores subsequent results | Q&A systems |
| **nDCG** | DCG / ideal_DCG | Handles graded relevance | Complex calculation | Recommendation systems |
| **ERR** | ∑(1/i ∏(1 - rel_j)) | Models user satisfaction | Less intuitive | Personalized ranking |

### Selection Guide
```
graph TD
    A[Use Case] --> B{User Behavior}
    B -->|Find first result| C[MRR]
    B -->|Explore multiple| D[nDCG]
    A --> E{Relevance Type}
    E -->|Binary| F[MAP]
    E -->|Graded| G[nDCG]
    A --> H{Result Set Size}
    H -->|Small fixed k| I[Precision@k]
    H -->|Variable| J[Recall@k]
```

# How does hybrid search work?

Hybrid search combines multiple retrieval methods to leverage their complementary strengths:

### Core Mechanism
```
graph LR
    A[Query] --> B(Keyword Search)
    A --> C(Vector Search)
    B --> D[Result Set A]
    C --> E[Result Set B]
    D & E --> F{Fusion Engine}
    F --> G[Final Ranked Results]
```

### Fusion Techniques
1. **Reciprocal Rank Fusion (RRF)**:
   ```
   score = ∑(1/(rank + k))
   k=60 (prevents domination by single method)
   ```
   
2. **Weighted Combination**:
   ```
   final_score = α•keyword_score + β•vector_score
   ```

3. **Round-Robin Interleaving**:
   ```
   Results: [Vec1, Key1, Vec2, Key2, ...]
   ```

### Component Roles
| **Method** | **Strength** | **Weakness** | **Weight Tuning** |
|------------|--------------|--------------|------------------|
| Keyword (BM25) | Precision on exact matches | Synonym handling | Higher for known-item search |
| Vector Search | Semantic understanding | Context fragmentation | Higher for exploratory queries |
| Metadata Filters | Constraint satisfaction | Over-constraining | Fixed weight based on field importance |

### Implementation Example
```
def hybrid_search(query):
    # Parallel retrieval
    keyword_results = bm25_search(query, top_k=50)
    vector_results = vector_search(query_embedding, top_k=50)
    
    # RRF fusion
    combined = {}
    for method, results in [('bm25', keyword_results), ('vec', vector_results)]:
        for rank, (doc_id, _) in enumerate(results):
            combined.setdefault(doc_id, 0)
            combined[doc_id] += 1/(60 + rank + 1)
    
    # Return sorted by fusion score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]
```

### Performance Benefits
1. **Recall Boost**:
   - Keyword finds exact matches
   - Vector finds semantic matches
   - Combined recall up to 98% vs 85% single-method

2. **Robustness**:
   - Handles both keyword-heavy and conceptual queries

3. **Practical Deployment**:
   ```
   Tools: Elasticsearch + dense_vector, Vespa, Weaviate
   Cloud: Azure Cognitive Search, AWS Kendra
   ```
# If you have search results from multiple methods, how would you merge and homogenize the rankings into a single result set?

Merging results from multiple retrieval methods requires sophisticated fusion techniques to create a unified ranking:

### Core Fusion Strategies
1. **Reciprocal Rank Fusion (RRF)**  
   ```
   score = ∑(1/(rank + k))  
   # k=60 prevents single-method domination
   ```
   - Preserves high-ranking items from all methods
   - Proven effectiveness in TREC competitions

2. **Weighted Combination**  
   ```
   final_score = α•keyword_score + β•vector_score + γ•metadata_score
   ```
   - Requires weight tuning via grid search:
     ```
     grid = {α: [0.3,0.5,0.7], β: [0.3,0.5,0.7], γ: [0.1,0.2]}
     best_weights = optimize(ndcg@10, grid)
     ```

3. **Round-Robin Interleaving**  
   ```
   results = [vec1, bm251, vec2, bm252, ...]
   ```
   - Ensures fair representation
   - Requires deduplication

### Advanced Homogenization Techniques
1. **Result Deduplication**  
   ```
   for result in all_results:
        if similar_to(existing, threshold=0.85):
            merge_scores(existing, result)
        else:
            add_new(result)
   ```

2. **Cross-Method Reranking**  
   - Train ML model to predict unified score:
     ```
     features = [bm25_score, vector_score, freshness, authority]
     unified_score = xgboost.predict(features)
     ```

3. **Diversity Enforcement**  
   ```
   while len(results) < k:
        next = highest_score(results)
        if cosine_sim(next, existing) < 0.7:
            add_to_final(next)
   ```

### Implementation Framework
```
def fuse_results(keyword_res, vector_res, meta_res):
    # RRF fusion
    combined = defaultdict(float)
    for method, results in [('kw', keyword_res), ('vec', vector_res)]:
        for rank, (doc_id, score) in enumerate(results):
            combined[doc_id] += 1/(60 + rank)
    
    # Apply metadata boost
    for doc_id in combined:
        if doc_id in meta_res:
            combined[doc_id] *= 1.2
    
    # Rerank with diversity
    return diversified_sort(combined, diversity_threshold=0.7)
```

### Evaluation Metrics for Fusion
1. **α-nDCG**: Measures diversity-aware ranking quality
2. **Intent-Aware Precision**: For multi-facet queries
3. **Novelty@k**: Percentage of unique aspects covered

# How to handle multi-hop/multifaceted queries?

Multi-hop queries require contextual reasoning across documents:

### Step-by-Step Processing
1. **Query Decomposition**  
   ```
   "Compare iPhone 15 and Samsung S23 cameras" →
   ["iPhone 15 camera specs", "Samsung S23 camera specs", "camera comparison reviews"]
   ```
   - Use LLM for decomposition:
     ```
     prompt = "Break into subqueries: {query}"
     subqueries = gpt4.generate(prompt)
     ```

2. **Sequential Retrieval**  
   ```
   context = []
   for subquery in subqueries:
        results = retrieve(subquery, context=context)
        context.append(summarize(results))
   ```

3. **Cross-Document Synthesis**  
   - Build knowledge graph:
     ```
     graph = Graph()
     for doc in results:
         entities = extract_entities(doc)
         graph.add_relations(entities)
     ```
   - Answer generation:
     ```
     answer = llm.generate(f"Based on {graph} answer {query}")
     ```

### Architecture for Multi-Hop Retrieval
```
graph TB
    A[Multi-hop Query] --> B(Query Decomposer)
    B --> C[Subquery 1]
    B --> D[Subquery 2]
    C --> E[Retriever 1]
    D --> F[Retriever 2]
    E --> G[Knowledge Graph Builder]
    F --> G
    G --> H[Answer Synthesizer]
    H --> I[Final Answer]
```

### Optimization Techniques
1. **Iterative Retrieval**  
   ```
   while not satisfied:
        new_query = llm.generate("Refine based on gap: {current_context}")
        results += retrieve(new_query)
   ```

2. **Evidence Chains**  
   - Store retrieval path:
     ```
     evidence = {
         "claim": "iPhone has better low-light",
         "sources": [doc123, doc456],
         "confidence": 0.92
     }
     ```

3. **Failure Recovery**  
   - Detect dead ends:
     ```
     if no_results(subquery):
         alternative = paraphrase(subquery)
         results = retrieve(alternative)
     ```

### Evaluation Framework
1. **Hop-Recall**: % of required subqueries addressed
2. **Path Accuracy**: Correctness of reasoning chain
3. **Source Diversity**: Coverage of different aspects

# What are different techniques to be used to improved retrieval?

Advanced retrieval enhancement techniques:

### 1. Query Expansion & Rewriting
- **Techniques**:
   ```
   # Synonym expansion
   expanded = query + " OR " + thesaurus[query]
   
   # LLM rewriting
   rewritten = gpt4.generate("Improve retrieval: {query}")
   ```
- **Tools**: Query Understanding Models (QUMs), spaCy

### 2. Negative Mining
- **Methods**:
   ```
   # Hard negative mining
   hard_negs = most_similar_but_irrelevant(query)
   
   # In-batch negatives
   negatives = random_from_same_batch
   ```
- **Impact**: Improves embedding space separation

### 3. Cross-Encoder Reranking
- **Workflow**:
   ```
   candidates = vector_retriever(query, top_k=100)
   reranked = cross_encoder.rerank(query, candidates)
   ```
- **Models**: MiniLM, Electra, DeBERTa

### 4. Temporal Adaptation
- **Strategies**:
   ```
   # Time-decay weighting
   score = original_score * exp(-age_days/30)
   
   # Freshness boost
   if doc_age < 7_days: score *= 1.5
   ```

### 5. Hybrid Indexing
- **Combination Methods**:
   | **Technique** | **Implementation** |
   |---------------|-------------------|
   | Vector+Keyword | RRF fusion |
   | Vector+Graph | Knowledge graph traversal |
   | Vector+Semantic | Entity-aware boosting |

### 6. Active Learning
- **Pipeline**:
   ```
   while accuracy < target:
        uncertain_queries = get_low_confidence_queries()
        human_label(uncertain_queries)
        retrain_model(new_data)
   ```

### 7. Domain Adaptation
- **Approaches**:
   ```
   # Domain-specific fine-tuning
   model.fit(medical_corpus, loss=TripletLoss())
   
   # Vocabulary augmentation
   tokenizer.add_tokens(["EGFR", "CT scan"])
   ```

### Performance Benchmarks
| **Technique** | **Recall@100 Gain** | **Latency Impact** |
|---------------|---------------------|-------------------|
| Query Expansion | +15% | +2ms |
| Cross-Encoder | +25% | +50ms |
| Hard Negatives | +12% | Training-only |
| Temporal Weighting | +8% (news) | Negligible |