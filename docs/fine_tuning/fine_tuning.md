# What is fine-tuning, and why is it needed?

Fine-tuning is the process of taking a pre-trained large language model (LLM) and further training it on a specialized dataset to adapt its capabilities to specific tasks or domains. Pre-trained LLMs like GPT or Llama are trained on massive general-domain datasets to develop broad linguistic understanding, but they lack specialized knowledge for particular applications.

Fine-tuning is needed for several reasons:
1. `Domain Adaptation`: Improves performance in specialized fields (medical, legal, technical) where generic models fail
2. `Task Specialization`: Adapts models to specific formats like question answering, summarization, or classification
3. `Style Alignment`: Adjusts output to match desired tone, formality level, or brand voice
4. `Bias Mitigation`: Reduces harmful outputs by retraining on curated datasets
5. `Vocabulary Expansion`: Incorporates domain-specific terminology not present in original training
6. `Efficiency Optimization`: Creates smaller, task-specific models that outperform larger generic models

Without fine-tuning, models often produce `hallucinations` (fabricated information) in specialized contexts and struggle with proprietary data formats.

# Which scenario do we need to fine-tune LLM?

Fine-tuning becomes essential in these scenarios:
1. `Domain-Specific Applications`: When working with specialized content (medical records, legal contracts, financial reports) requiring precise terminology
2. `Output Format Constraints`: When responses must follow strict templates (API responses, structured data extraction)
3. `Consistency Requirements`: For applications demanding uniform response patterns across thousands of queries
4. `Proprietary Knowledge Integration`: When incorporating confidential data not suitable for public API access
5. `Low-Latency Systems`: When deploying optimized models to edge devices with resource constraints
6. `Hallucination-Sensitive Domains`: In high-risk fields (healthcare diagnostics, engineering) where factual accuracy is critical
7. `Regulatory Compliance`: When outputs must adhere to industry-specific regulations (HIPAA, GDPR)

Common use cases: customer support bots with brand-specific protocols, academic paper analysis tools, and technical documentation assistants.

# How to make the decision of fine-tuning?

Follow this 4-step decision framework:
1. `Problem Diagnosis`:
   - Test base model performance using `prompt engineering` techniques (few-shot learning, chain-of-thought)
   - Quantify performance gaps with metrics like `F1-score` or `BLEU` (<75% indicates need)
   - Verify sufficient quality data exists (`500-10k curated examples` required)
2. `Alternative Evaluation`:
   - Benchmark `RAG` (Retrieval-Augmented Generation) approach first
   - Test smaller specialized models before large-scale fine-tuning
   - Evaluate cost of API solutions vs in-house deployment
3. `Requirement Validation`:
   - Confirm domain specificity needs non-public knowledge
   - Verify response format demands template adherence
   - Validate latency constraints (`<500ms` requires optimization)
4. `Cost-Benefit Analysis`:
   - Calculate infrastructure costs (`GPU hours × cloud rates`)
   - Estimate annotation/curation expenses (`$3-10` per example)
   - Project accuracy improvement ROI (+15% typically justifies investment)

Fine-tune when all of these converge: performance gap >15%, data availability confirmed, and task specificity exceeds prompt engineering capabilities.

# How do you improve the model to answer only if there is sufficient context for doing so?

Implement these techniques to create `context-sensitive models`:
1. `Data Engineering`:
   - Add `negative examples` to training data: 15-20% unanswerable questions with `[No sufficient context]` labels
   - Use `context-question mismatches`: Pair irrelevant context snippets with specific questions
   - Generate synthetic data using `confidence-aware augmentation` tools like `RAGAs`
2. `Architecture Modifications`:
   - Add `rejection head`: Secondary classifier predicting answerability (uses `CLS token embeddings`)
   - Implement `confidence thresholds`: Discard responses when softmax probability <0.75
   - Use `perplexity filters`: Block outputs with high token uncertainty (>150 perplexity)
3. `Training Techniques`:
   - Custom `loss function`: L = L_CE + λ × `KL-divergence`(confidence_predictions, confidence_labels)
   - `Multi-task learning`: Jointly train for question answering and answerability classification
4. `Inference Controls`:
   - `Cosine similarity checks`: Reject when query-context similarity <0.8
   - `Minimal token enforcement`: Return `INSUFFICIENT_CONTEXT` for answers <5 tokens to complex queries
   - `Context-length thresholds`: Require minimum context characters per word in question
5. `Prompt Engineering`:
   - System message: `Always respond `[Not enough context]` when information is missing or contradictory`
   - Output constraints: `Enforce JSON format with `confidence_score` and `rejection_flag` fields`

# How to create fine-tuning datasets for Q&A?

Creating high-quality fine-tuning datasets for question answering requires a structured, multi-stage approach:

### 1. `Source Identification and Collection`
- **Document Sources**:
  - Internal resources: `Product documentation`, `knowledge bases`, `support ticket histories` (PII redacted)
  - External materials: `Technical whitepapers`, `academic publications`, `regulatory guidelines`
  - Domain-specific content: `Medical journals`, `legal case files`, `financial reports`
- **Synthetic Generation**:
  - Use `GPT-4` or `Claude` with prompt engineering:
    ```
    Generate 10 QA pairs from this context: [Your Text Here]
    Include 2 unanswerable questions
    ```
  - Leverage specialized tools:
    - `RAGAs` for context-aware augmentation
    - `Anthropic's Constitutional AI` for safety-aligned examples
- **Public Datasets** (for transfer learning):
  - `SQuAD v2` (100k+ Wikipedia QA pairs)
  - `Natural Questions` (real user questions with Wikipedia answers)
  - `HotpotQA` (multi-hop reasoning)

### 2. `Data Construction Methodology`
- **Core QA Triplet Structure**:
  ```
  {
    `id`: `q_12345`,
    `context`: `iPhone 15 Pro features titanium chassis...`,
    `question`: `What material is used in iPhone 15 Pro?`,
    `answer`: `titanium`
  }
  ```
- **Essential Components**:
  - `70% answerable` questions
  - `20% unanswerable` questions (with `[No context]` answer)
  - `10% adversarial` questions (misleading but context-related)
- **Multi-hop QA** (15-20% of dataset):
  - Require synthesis across documents
  - Example: 
    ```
    Context1: `John works at Google`
    Context2: `Google HQ is in Mountain View`
    Question: `Where does John work?`
    Answer: `Mountain View`
    ```

### 3. `Advanced Augmentation Techniques`
- **Paraphrase Generation**:
  - Tools: `T5 paraphrases`, `Pegasus`, `Sentence Transformers`
  - Input: `How to reset password?` → Outputs:
    1. `What's the password recovery procedure?`
    2. `Steps for resetting login credentials?`
- **Entity Swapping**:
  - `Product-based`: Dell XPS → Lenovo ThinkPad
  - `Location-based`: New York → Singapore
  - `Numerical variations`: 5GB → 10GB storage
- **Context Distortion**:
  - Random sentence removal (15-30% deletion)
  - Insertion of contradictory statements
  - Cross-document mixing

### 4. `Quality Control Pipeline`
- **Automated Filtering**:
  ```
  # Deduplication with MinHash
  from datasketch import MinHashLSH
  lsh = MinHashLSH(threshold=0.7, num_perm=128)
  ```
- **Semantic Validation**:
  - Calculate `BERT similarity` between question and context (filter <0.3)
  - Check answer span with `regular expressions`
- **Human Annotation**:
  | Checkpoint | Criteria | Rejection Rate |
  |------------|----------|---------------|
  | `Round 1` | Answer correctness | 20-30% |
  | `Round 2` | Context relevance | 10-15% |
  | `Adjudication` | Expert review | 5-10% |
- **PII Handling**:
  - Use `Presidio` or `Amazon Comprehend` for automatic redaction
  - Replace with `[REDACTED]` or synthetic equivalents

### 5. `Dataset Optimization`
- **Answerability Balance**:
  - Ideal distribution:
    ```
    Answerable: 65-75%
    Unanswerable: 20-25%
    Ambiguous: 5-10%
    ```
- **Difficulty Grading**:
  - Simple fact retrieval: 40%
  - Inferential reasoning: 35%
  - Multi-hop synthesis: 25%
- **Token Length Analysis**:
  - Questions: `8-25 tokens`
  - Contexts: `128-1024 tokens`
  - Answers: `5-50 tokens`

### 6. `Format Conversion and Versioning`
- **Standard Formats**:
  - `JSON Lines` (for OpenAI/Hugging Face):
    ```
    {"prompt": "Context: [text]`n`Question: [query]", "completion": "[answer]"}
    ```
  - `SQuAD-style`:
    ```
    "qas": [{
        "question": "...",
        "answers": [{"text": "...", "answer_start": 42}]
    }]
    ```
- **Version Control**:
  - Use `DVC` (Data Version Control) with `S3/GCS` storage
  - Maintain `data cards` documenting:
    - Provenance
    - Annotation methodology
    - Update history

### 7. `Dataset Scaling Guidelines`
| Application | Minimum Size | Ideal Size | Cost Estimate |
|-------------|--------------|------------|---------------|
| `Proof-of-Concept` | 500 QA pairs | 1,000 | $1,500-$3,000 |
| `Production MVP` | 5,000 | 15,000 | $12,000-$25,000 |
| `Enterprise System` | 20,000 | 100,000+ | $75,000-$200,000 |


# How to set hyperparameters for fine-tuning?

Hyperparameter optimization requires balancing performance, training stability, and resource constraints. Use this evidence-based framework:

### 1. `Learning Rate (LR) Strategy`
- **Cyclical Scheduling**: 
  - Base LR: `3e-5` → Peak: `1e-4` → Final: `5e-6`
  - Implement via ``
    from torch.optim.lr_scheduler import CyclicLR
    scheduler = CyclicLR(optimizer, base_lr=3e-5, max_lr=1e-4)
    ```
- **LR Finder Protocol**:
  - Exponentially increase LR from `1e-7` until loss spikes (`typically at 1e-3`)
  - Optimal LR = `0.5× spike point` (`Smith 2017 method`)

### 2. `Batch Optimization`
- **Memory-Calibrated Sizing**:
  - Formula: `max_batch = floor(GPU_VRAM / (params * precision_factor))`
  - Precision factors: 
    - fp32: `18` 
    - fp16: `10` 
    - int8: `6`
- **Gradient Handling**:
  - Accumulation: `steps = desired_batch / max_batch` (typ. `2-16`)
  - Clip norms: `global_norm=1.0` (`prevents explosion`)

### 3. `Regularization Configuration`
| Technique | Small Data (<5k) | Large Data (>50k) | Implementation |
|-----------|------------------|-------------------|----------------|
| `Dropout` | `0.3-0.5` | `0.1` | ``
    nn.Dropout(p=0.3)
    ``` |
| `Weight Decay` | `0.01` | `0.05` | ``
    AdamW(weight_decay=0.01)
    ``` |
| `Label Smoothing` | `0.2` | `0.1` | ``
    CrossEntropyLoss(label_smoothing=0.1)
    ``` |

### 4. `Duration Controls`
- **Epoch Management**:
  - Rule: `1 epoch per 1k samples` (max `10 epochs`)
  - Early stopping: `patience=4` epochs with `min_delta=0.005`
- **Checkpointing**:
  - Save interval: `1000 steps` OR `0.3 epoch`
  - Metric: `validation loss` + `task-specific accuracy`

### 5. `Advanced Calibration`
- **Precision Selection**:
  ``
  # Automatically select best precision
  precision = `bf16` if torch.cuda.has_bf16_support() else `fp16`
  ```
- **Hyperparameter Search**:
  ``
  optuna.study(
      lr: [1e-5, 3e-5, 1e-4],
      batch_size: [4, 8, 16]
  )
  ```
  (Run `100 trials` with TPE sampler)

# How to estimate infrastructure requirements for fine-tuning LLM?

Use these quantitative estimation techniques:

### 1. `Core Resource Formulas`
| Resource | Calculation | Constants |
|----------|-------------|----------|
| `VRAM (GB)` | `(P × 20) × safety(1.5)` | fp16: `20 bytes/param` |
| `Training Time (hrs)` | `(6P × S × L) / (GPU_flops × util)` | A100: `312 TFLOPS`, util:`0.4` |
| `Storage (TB)` | `(P × 2 × C) + (D × 5)` | Checkpoints:`10`, Data factor:`5` |

Where:
- `P` = Parameters (billions)
- `S` = Training steps
- `L` = Sequence length
- `C` = Checkpoint count
- `D` = Dataset size (GB)

### 2. `Real-World Estimation Table`
| Model Size | Min VRAM | GPU Config | Training Time* | Cloud Cost** |
|------------|----------|------------|----------------|-------------|
| `7B` | `140GB` | `2× A100 80GB` | `48hrs` | `$2300` |
| `13B` | `260GB` | `4× A100 80GB` | `72hrs` | `$6200` |
| `70B` | `1.4TB` | `16× A100 80GB` | `240hrs` | `$58k` |

*For 1M samples @ seqlen=2048  
**Based on $3.67/hr per A100 80GB

### 3. `Multi-GPU Strategies`
- **Data Parallelism**: Linear scaling (`good for <=4 GPUs`)
- **Model Parallelism**:
  - ZeRO Stage 3: `8× VRAM reduction` (high comms cost)
  - Pipeline Parallel: Layer splitting (`chunk_size=4` optimal)
- **Hybrid Approach**: 
  ``
  DeepSpeedConfig(
      zero_optimization=`stage:3`,
      pipeline=`enabled:true`
  )
  ```

# How do you fine-tune LLM on consumer hardware?

Employ these hardware-constrained methodologies:

### 1. `Core Techniques`
| Method | VRAM Reduction | Implementation |
|--------|----------------|----------------|
| `QLoRA` | `4×` | ``
    model = AutoModelForCausalLM.from_pretrained(
        加载在4位=True, 
        bnb_4bit_compute_dtype=`torch.bfloat16`
    )
    ``` |
| `Gradient Checkpointing` | `65%` | ``
    model.gradient_checkpointing_enable()
    ``` |
| `CPU Offloading` | `10-20×` | Use `bitsandbytes` 8-bit optimizer |
| `Int8 Inference` | `2×` | ``
    model.quantize(`bits:8`)
    ``` |

### 2. `Hardware Profiles`
| Configuration | Max Model | Throughput | Tools |
|---------------|----------|------------|-------|
| RTX 4090 (24GB) | `13B` | `20 tok/s` | `QLoRA` + `FlashAttention` |
| Dual RTX 3090 | `20B` | `14 tok/s` | `DeepSpeed ZeRO-2` |
| M2 Ultra (192GB) | `30B` | `8 tok/s` | `llama.cpp` + `CPU offload` |
| RTX 3060 (12GB) | `7B` | `6 tok/s` | `4-bit quantization` |

### 3. `Optimized Training Loop`
```

from peft import LoraConfig

peft_config = LoraConfig(

r=`8`,

lora_alpha=`32`,

target_modules=`["q_proj","v_proj"]`

)

trainer = SFTTrainer(

model,

train_dataset,

peft_config=peft_config,

args=TrainingArguments(

per_device_train_batch_size=`2`,

gradient_accumulation_steps=`8`,

fp16=`True`

)

)

```

# What are the different categories of the PEFT method?

### `Taxonomy of Efficiency`
1. **Additive Methods**
   - `Soft Prompting`: Train prefix vectors (`100-500 tokens`)
   - `Adapters`: Add FFN layers between transformers
     - Variants: `Houlsby` (parallel), `Compacter` (low-rank)

2. **Reparameterized Methods**
   - `Low-Rank Approximations` (LoRA, AdaLoRA)
   - `Weight Subspace Methods` (DiffPruning)

3. **Selective Methods**
   - `Layer Freezing`: Update only `last N layers`
   - `Bias-Only Tuning` (BitFit: `0.1% params`)
   - `Attention Head Masking`

4. **Hybrid Techniques**
   - `LoRA+Adapter`: Combines injection + decomposition
   - `QLoRA`: `4-bit quantized LoRA`
   - `IA³`: `Inhibit/Amplify activations`

### `Method Comparison Table`
| Method | Parameters | Memory | Performance |
|--------|------------|--------|-------------|
| `LoRA` | `0.5-5%` | Low | `95-99% FT` |
| `Prefix-Tuning` | `<0.1%` | Very Low | `85-92% FT` |
| `Adapters` | `1-4%` | Medium | `90-96% FT` |
| `QLoRA` | `0.5%` | Ultra Low | `94-97% FT` |

# What is catastrophic forgetting in LLMs?

### `Mechanism Analysis`
Catastrophic forgetting occurs due to:
1. **Parameter Overwriting**  
   New task gradients overwrite `general knowledge weights` in early layers
   
2. **Attention Distortion**  
   Specialization alters `attention distributions` (Layer `12/40` most vulnerable)

3. **Representation Collapse**  
   High-dimensional manifolds collapse into `task-specific subspaces`

### `Quantifiable Impact`
When fine-tuning on new domain (`10k samples`):
- `CoLA` score drops 38% (language acceptability)
- `Natural Questions` accuracy reduced 52%
- Hallucination rate increases `25-60%`

### `Mitigation Framework`
1. **Elastic Regularization**  
   ``
   EWC_loss = sum(λ * F_i * (θ_i - θ*_i)^2)
   ```  
   (`F` = Fisher information matrix)

2. **Memory Buffers**  
   Store `10% original data` → `periodic replay`

3. **Architectural Isolation**  
   - Lateral connection gates (`Progressive Nets`)
   - Task-specific residual adapters

4. **Curriculum Strategies**  
   Joint training with decaying old-task weighting:  
   `λ = 0.5 → λ = 0.1 over epochs`

# What are different re-parameterized methods for fine-tuning?

### `Low-Rank Matrix Decomposition`
1. **Standard LoRA**  
   - Formula: ΔW = A · B^T (A ∈ ℝ^{d×r}, B ∈ ℝ^{k×r})  
   - Optimal rank: `r=8` (`` ablation studies)
   - Implementation:  
     ``
     class LoRALayer(nn.Module):
         def __init__(self, r=8):
             self.A = nn.Linear(d, r, bias=False)
             self.B = nn.Linear(r, k, bias=False)
     ```

2. **AdaLoRA**  
   - Adaptive rank allocation via `SVD thresholding`
   - Advantages: `4× parameter efficiency`

3. **LyRA**  
   - Block-sparse formulation:  
     `ΔW = Σ S_i · T_i` (sparse blocks)
   - VRAM reduction: `10× over full finetuning`

### `Factorized Methods`
1. **TENSOR FACTORIZATION**  
   - Tucker decomposition: W = G × U × V × T  
   - Compression: `15×` for FFN layers

2. **INTRINSIC SAID**  
   - Random projection dictionaries:  
     ``
     D = random_matrix(n, r)
     ΔW = D · C
     ```

### `Weight Subspace Techniques`
1. **DIFFPRUNING**  
   - Sparse gradient mask via `ℓ0 regularization`:  
     ``
     L = L_task + λ||θ - θ₀||₀
     ```

2. **MOVEMENT PRUNING**  
   - Learns weight importance scores:  
     `I_ij = |θ_ij - θ⁰_ij|`  
     Prunes `50-80% parameters`