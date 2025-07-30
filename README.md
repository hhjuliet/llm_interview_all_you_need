[中文](#section-zh) | [English](#section-en)

<a name="section-zh"></a>

<!-- 中英文分栏布局 -->

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div style="grid-column: 1;">

<!-- 中文部分 -->

# LLM 面试完全指南 (llm_interview_all_you_need)

📚 收集顶级公司 LLM 面试高频问题，欢迎社区共创！
🚀 正在不断整理来自 Google/OpenAI/Meta/Anthropic 等公司的真题，涵盖 RAG、微调、部署等核心领域。

## 目录

1. [提示工程与 LLM 基础](#prompt-engineering--basics-of-llm)
2. [检索增强生成 (RAG)](#retrieval-augmented-generation-rag)
3. [文档数字化与分块](#document-digitization--chunking)
4. [嵌入模型](#embedding-models)
5. [向量数据库原理](#internal-working-of-vector-databases)
6. [高级搜索算法](#advanced-search-algorithms)
7. [语言模型原理](#language-models-internal-working)
8. [监督微调 (SFT)](#supervised-fine-tuning-of-llm)
9. [偏好对齐 (RLHF/DPO)](#preference-alignment-rlhfdpo)
10. [评估 LLM 系统](#evaluation-of-llm-system)
11. [幻觉控制技术](#hallucination-control-techniques)
12. [LLM 部署方案](#deployment-of-llm)
13. [智能体系统](#agent-based-system)
14. [提示注入攻防](#prompt-hacking)
15. [综合话题](#miscellaneous)
16. [实战案例](#case-studies)

---

### 提示工程与 LLM 基础

<details>
<summary>查看问题</summary>

- **生成式 AI 与判别式 AI 的核心区别？**
- **语言模型的训练流程解析**
- **Temperature 参数的作用与设置原则**
- **LLM 解码策略比较分析**
- **如何定义大语言模型的停止条件？**
- **停止序列在 LLM 中的应用方法**
- **提示工程的基本结构是什么？**
- **上下文学习机制解析**
- **提示工程的类型与实施方法**
- **少样本提示的关键注意事项**
- **编写高质量提示的有效策略**
- **如何通过提示工程控制 LLM 幻觉**
- **使用提示工程增强 LLM 推理能力**
- **当思维链(CoT)提示失效时的改进方法**

[查看完整文档 →](/docs/prompt_engineering_basics)

</details>

### 检索增强生成 (RAG)

<details>
<summary>查看问题</summary>

- **如何提高 LLM 输出的准确性和可靠性**
- **RAG 工作机制详细解析**
- **使用 RAG 系统的主要优势**
- **微调 vs RAG 的选择标准**
- **私有数据定制化 LLM 的架构模式**

[查看完整文档 →](/docs/rag_systems)

</details>

### 文档数字化与分块

<details>
<summary>查看问题</summary>

- **分块的基本概念与必要性**
- **影响分块大小的关键因素**
- **不同类型的分块方法比较**
- **寻找最佳分块大小的策略**
- **复杂文档（年报）的数字化处理方案**
- **表格处理的最佳实践**
- **大型表格的检索优化方法**
- **列表项的分块处理技术**
- **生产级文档处理流水线构建**
- **RAG 系统中的图表处理方案**

[查看完整文档 →](/docs/document_processing)

</details>

### 嵌入模型

<details>
<summary>查看问题</summary>

- **向量嵌入与嵌入模型的基本概念**
- **LLM 应用中嵌入模型的使用场景**
- **长短内容嵌入的区别与优化**
- **如何基于私有数据评测嵌入模型**
- **OpenAI 嵌入模型精度不足的优化方案**
- **改进 Sentence Transformer 模型的步骤**

[查看完整文档 →](/docs/embedding_models)

</details>

### 向量数据库原理

<details>
<summary>查看问题</summary>

- **向量数据库的基本原理**
- **向量数据库与传统数据库的差异**
- **索引、数据库与插件的区别**
- **高精度搜索场景下的策略选择**
- **聚类与局部敏感哈希等搜索策略**
- **聚类减少搜索空间的机制**
- **随机投影索引工作原理**
- **局部敏感哈希(LSH)实现机制**
- **乘积量化(PQ)索引方法**
- **不同向量索引的场景应用比较**
- **相似度度量的选择标准**
- **向量数据库过滤的挑战与解决方案**
- **向量数据库选型指南**

[查看完整文档 →](/docs/vector_databases)

</details>

### 高级搜索算法

<details>
<summary>查看问题</summary>

- **信息检索与语义搜索的架构模式**
- **高质量搜索系统的重要性**
- **大规模数据集的高效精准搜索**
- **改进不准确 RAG 检索系统的步骤**
- **基于关键词的检索方法**
- **优化重排模型的微调技术**
- **信息检索常用指标及局限性**
- **类 Quora 系统的评价指标选择**
- **推荐系统的评价指标**
- **不同信息检索指标的应用场景**
- **混合搜索的工作原理**
- **多源搜索结果的合并策略**
- **多轮查询的处理技术**
- **改进检索效果的高级技术**

[查看完整文档 →](/docs/search_algorithms)

</details>

### 语言模型原理

<details>
<summary>查看问题</summary>

- **自注意力机制的详细解析**
- **自注意力的缺陷与改进方案**
- **位置编码的工作原理**
- **Transformer 架构深度剖析**
- **Transformer 相对 LSTM 的优势**
- **局部注意力和全局注意力的区别**
- **Transformer 计算资源消耗优化**
- **扩展 LLM 上下文长度的技术**
- **大词表下的架构优化方案**
- **词表大小平衡策略**
- **不同 LLM 架构的适用场景**

[查看完整文档 →](/docs/llm_internals)

</details>

### 监督微调 (SFT)

<details>
<summary>查看问题</summary>

- **微调的概念与必要性**
- **需要微调的场景分析**
- **微调决策的评估流程**
- **基于上下文的精确回答优化**
- **QA 微调数据集构建方法**
- **微调超参数设置指南**
- **微调基础设施需求估算**
- **消费级硬件的微调方案**
- **参数高效微调(PEFT)方法分类**
- **灾难性遗忘问题解析**
- **重参数化微调方法**

[查看完整文档 →](/docs/fine_tuning)

</details>

### 偏好对齐 (RLHF/DPO)

<details>
<summary>查看问题</summary>

- **选择偏好对齐方法的时机**
- **RLHF 的工作机制与应用**
- **RLHF 中的奖励黑客问题**
- **不同偏好对齐方法比较**

[查看完整文档 →](/docs/preference_alignment)

</details>

### 评估 LLM 系统

<details>
<summary>查看问题</summary>

- **如何评估最适合的 LLM 模型**
- **RAG 系统评估方法论**
- **LLM 评估指标大全**
- **验证链(Chain of Verification)解析**

[查看完整文档 →](/docs/llm_evaluation)

</details>

### 幻觉控制技术

<details>
<summary>查看问题</summary>

- **不同形式的幻觉分类**
- **多层次幻觉控制技术**

[查看完整文档 →](/docs/hallucination_control)

</details>

### LLM 部署方案

<details>
<summary>查看问题</summary>

- **量化不影响精度的原理**
- **LLM 推理吞吐量优化技术**
- **无注意力近似的响应加速方案**

[查看完整文档 →](/docs/llm_deployment)

</details>

### 智能体系统

<details>
<summary>查看问题</summary>

- **智能体基本概念与实现策略**
- **智能体的需求与常见架构**
- **ReAct 提示实现示例**
- **计划与执行策略详解**
- **OpenAI 函数使用实例**
- **OpenAI 函数 vs LangChain 智能体**

[查看完整文档 →](/docs/agent_systems)

</details>

### 提示注入攻防

<details>
<summary>查看问题</summary>

- **提示攻击的基本概念与危害**
- **不同类型提示攻击分析**
- **防御提示攻击的策略**

[查看完整文档 →](/docs/prompt_hacking)

</details>

### 综合话题

<details>
<summary>查看问题</summary>

- **LLM 系统成本优化方案**
- **专家混合模型(MoE)解析**
- **生产级 RAG 系统构建指南**
- **FP8 变量及其优势**
- **无损精度低精度训练技术**
- **KV 缓存大小计算方法**
- **多头注意力层维度分析**
- **注意力层焦点控制技术**

[查看完整文档 →](/docs/miscellaneous)

</details>

### 实战案例

<details>
<summary>查看问题</summary>
敬请期待
</details>

### 🤝 如何贡献

1. 提交新问题到对应分类的 `.md` 文件
2. 完善现有问题答案（需标注引用来源）
3. 改进文档结构或翻译
4. 欢迎提交真实面试经历！

</div>
<div style="grid-column: 2;">
[返回顶部↑](#section-zh)


<a name="section-en"></a>

<!-- 英文部分 -->

# LLM Interview All You Need

📚 Curating top company LLM interview questions with community collaboration!
🚀 Continuously updating real questions from Google/OpenAI/Meta/Anthropic covering RAG, fine-tuning, deployment and more.

## Table of Contents

1. [Prompt Engineering & LLM Basics](#prompt-engineering--basics-of-llm)
2. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
3. [Document Digitization & Chunking](#document-digitization--chunking)
4. [Embedding Models](#embedding-models)
5. [Vector DB Internals](#internal-working-of-vector-databases)
6. [Advanced Search Algorithms](#advanced-search-algorithms)
7. [Language Model Internals](#language-models-internal-working)
8. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-of-llm)
9. [Preference Alignment (RLHF/DPO)](#preference-alignment-rlhfdpo)
10. [Evaluating LLM Systems](#evaluation-of-llm-system)
11. [Hallucination Control](#hallucination-control-techniques)
12. [LLM Deployment Strategies](#deployment-of-llm)
13. [Agent-Based Systems](#agent-based-system)
14. [Prompt Hacking Defenses](#prompt-hacking)
15. [Miscellaneous Topics](#miscellaneous)
16. [Case Studies](#case-studies)

---

### Prompt Engineering & Basics

<details>
<summary>View Questions</summary>

- **What is the difference between Predictive/Discriminative AI and Generative AI?**
- **What is LLM, and how are LLMs trained?**
- **Explain the Temperature parameter and how to set it**
- **What are different decoding strategies for tokens?**
- **Stopping criteria definition methods**
- **How to use stop sequences in LLMs?**
- **Basic structure of prompt engineering**
- **Explain in-context learning**
- **Types of prompt engineering**
- **Aspects to keep in mind with few-shots prompting**
- **Strategies to write effective prompts**
- **Hallucination control via prompts**
- **Improving reasoning through prompts**
- **Solution when COT prompt fails**

[View Full Document →](/docs/prompt_engineering_basics)

</details>

### Retrieval Augmented Generation

<details>
<summary>View Questions</summary>

- **Increase accuracy & reliability in LLM**
- **How RAG works?**
- **Benefits of RAG systems**
- **Fine-tuning vs RAG selection**
- **LLM customization patterns**

[View Full Document →](/docs/rag_systems)

</details>

### Document Digitization & Chunking

<details>
<summary>View Questions</summary>

- **What is chunking and why needed**
- **Factors influencing chunk size**
- **Different chunking methods**
- **Finding ideal chunk size**
- **Digitizing complex documents**
- **Handling tables during chunking**
- **Retrieval optimization for large tables**
- **List item processing techniques**
- **Production document pipelines**
- **Handling graphs & charts in RAG**

[View Full Document →](/docs/document_processing)

</details>

### Embedding Models

<details>
<summary>View Questions</summary>

- **Vector embeddings basics**
- **Embedding model usage in LLM apps**
- **Short vs long content embedding**
- **Benchmarking embedding models**
- **Improving low-accuracy embedding model**
- **Enhancing sentence transformers**

[View Full Document →](/docs/embedding_models)

</details>

### Vector DB Internals

<details>
<summary>View Questions</summary>

- **What is a vector database?**
- **How does a vector database differ from traditional databases?**
- **How does a vector database work?**
- **Explain difference between vector index, vector DB & vector plugins?**
- **You are working on a project that involves a small dataset of customer reviews. Your task is to find similar reviews in the dataset. The priority is to achieve perfect accuracy in finding the most similar reviews, and the speed of the search is not a primary concern. Which search strategy would you choose and why?**
- **Explain vector search strategies like clustering and Locality-Sensitive Hashing.**
- **How does clustering reduce search space? When does it fail and how can we mitigate these failures?**
- **Explain Random projection index?**
- **Explain Locality-sensitive hashing (LHS) indexing method?**
- **Explain product quantization (PQ) indexing method?**
- **Compare different Vector index and given a scenario, which vector index you would use for a project?**
- **How would you decide ideal search similarity metrics for the use case?**
- **Explain different types and challenges associated with filtering in vector DB?**
- **How to decide the best vector database for your needs?**

[View Full Document →](/docs/vector_databases)

</details>

### Advanced Search Algorithms

<details>
<summary>View Questions</summary>

- **What are architecture patterns for information retrieval & semantic search?**
- **Why it’s important to have very good search**
- **How can you achieve efficient and accurate search results in large-scale datasets?**
- **Consider a scenario where a client has already built a RAG-based system that is not giving accurate results, upon investigation you find out that the retrieval system is not accurate, what steps you will take to improve it?**
- **Explain the keyword-based retrieval method**
- **How to fine-tune re-ranking models?**
- **Explain most common metric used in information retrieval and when it fails?**
- **If you were to create an algorithm for a Quora-like question-answering system, with the objective of ensuring users find the most pertinent answers as quickly as possible, which evaluation metric would you choose to assess the effectiveness of your system?**
- **I have a recommendation system, which metric should I use to evaluate the system?**
- **Compare different information retrieval metrics and which one to use when?**
- **How does hybrid search works?**
- **If you have search results from multiple methods, how would you merge and homogenize the rankings into a single result set?**
- **How to handle multi-hop/multifaceted queries?**
- **What are different techniques to be used to improved retrieval?**

[View Full Document →](/docs/search_algorithms)

</details>

### LLM Internals

<details>
<summary>View Questions</summary>

- **Can you provide a detailed explanation of the concept of self-attention?**
- **Explain the disadvantages of the self-attention mechanism and how can you overcome it.**
- **What is positional encoding?**
- **Explain Transformer architecture in detail.**
- **What are some of the advantages of using a transformer instead of LSTM?**
- **What is the difference between local attention and global attention?**
- **What makes transformers heavy on computation and memory, and how can we address this?**
- **How can you increase the context length of an LLM?**
- **If I have a vocabulary of 100K words/tokens, how can I optimize transformer architecture?**
- **A large vocabulary can cause computation issues and a small vocabulary can cause OOV issues, what approach you would use to find the best balance of vocabulary?**
- **Explain different types of LLM architecture and which type of architecture is best for which task?**

[View Full Document →](/docs/llm_internals)

</details>

### Supervised Fine-Tuning

<details>
<summary>View Questions</summary>

- **What is fine-tuning, and why is it needed?**
- **Which scenario do we need to fine-tune LLM?**
- **How to make the decision of fine-tuning?**
- **How do you improve the model to answer only if there is sufficient context for doing so?**
- **How to create fine-tuning datasets for Q&A?**
- **How to set hyperparameters for fine-tuning?**
- **How to estimate infrastructure requirements for fine-tuning LLM?**
- **How do you fine-tune LLM on consumer hardware?**
- **What are the different categories of the PEFT method?**
- **What is catastrophic forgetting in LLMs?**
- **What are different re-parameterized methods for fine-tuning?**

[View Full Document →](/docs/fine_tuning)

</details>

### Preference Alignment

<details>
<summary>View Questions</summary>

- **At which stage you will decide to go for the Preference alignment type of method rather than SFT?**
- **What is RLHF, and how is it used?**
- **What is the reward hacking issue in RLHF?**
- **Explain different preference alignment methods.**

[View Full Document →](/docs/preference_alignment)

</details>

### Evaluating LLM Systems

<details>
<summary>View Questions</summary>

- **How do you evaluate the best LLM model for your use case?**
- **How to evaluate RAG-based systems?**
- **What are different metrics for evaluating LLMs?**
- **Explain the Chain of Verification.**

[View Full Document →](/docs/llm_evaluation)

</details>

### Hallucination Control

<details>
<summary>View Questions</summary>

- **What are different forms of hallucinations?**
- **How to control hallucinations at various levels?**

[View Full Document →](/docs/hallucination_control)

</details>

### LLM Deployment

<details>
<summary>View Questions</summary>

- **Why quantization preserves accuracy**
- **LLM inference throughput optimization**
- **Accelerating response time**

[View Full Document →](/docs/llm_deployment)

</details>

### Agent-Based Systems

<details>
<summary>View Questions</summary>

- **Agent concepts & strategies**
- **Need for agents & implementation**
- **ReAct prompting example**
- **Plan and Execute strategy**
- **OpenAI functions examples**
- **OpenAI vs LangChain Agents**

[View Full Document →](/docs/agent_systems)

</details>

### Prompt Hacking

<details>
<summary>View Questions</summary>

- **Prompt hacking explained**
- **Types of prompt hacking**
- **Defense tactics**

[View Full Document →](/docs/prompt_hacking)

</details>

### Miscellaneous

<details>
<summary>View Questions</summary>

- **Optimizing LLM system cost**
- **Mixture of Expert (MoE) models**
- **Building production RAG system**
- **FP8 advantages**
- **Low precision training**
- **KV cache size calculation**
- **Attention layer dimensions**
- **Focusing attention layers**

[View Full Document →](/docs/miscellaneous)

</details>

### Case Studies

<details>
<summary>View Questions</summary>

Coming soon

</details>

### 🤝 How to Contribute

1. Add new questions to corresponding `.md` files
2. Improve existing answers (with citations)
3. Enhance documentation structure or translations
4. Share real interview experiences!

</div>
</div>
[Back to Top↑](#section-en)



[Back to Top↑](#section-en)## 授权协议 (License)

版权所有 (c) 2025 llm_interview_all_you_need

特此授予任何获得本软件副本及相关文档文件（以下简称"软件"）的任何人免费许可，允许其不受限制地处理本软件，包括但不限于使用、复制、修改、合并、发布、分发、再授权及/或销售软件副本的权利，并允许获得软件的人这样做，但须满足以下条件：

上述版权声明和本许可声明应包含在软件的所有副本或实质性部分中。

本软件按"原样"提供，不提供任何明示或暗示的保证，包括但不限于适销性保证、特定用途适用性保证和非侵权保证。在任何情况下，作者或版权所有者均不对因软件或软件使用或其他交易行为而产生的任何索赔、损害或其他责任负责。

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
<div style="grid-column: 1;">

<!-- 中文部分 -->

# LLM 面试完全指南 (llm_interview_all_you_need)

📚 收集顶级公司 LLM 面试高频问题，欢迎社区共创！
🚀 正在不断整理来自 Google/OpenAI/Meta/Anthropic 等公司的真题，涵盖 RAG、微调、部署等核心领域。

## 目录

1. [提示工程与 LLM 基础](#prompt-engineering--basics-of-llm)
2. [检索增强生成 (RAG)](#retrieval-augmented-generation-rag)
3. [文档数字化与分块](#document-digitization--chunking)
4. [嵌入模型](#embedding-models)
5. [向量数据库原理](#internal-working-of-vector-databases)
6. [高级搜索算法](#advanced-search-algorithms)
7. [语言模型原理](#language-models-internal-working)
8. [监督微调 (SFT)](#supervised-fine-tuning-of-llm)
9. [偏好对齐 (RLHF/DPO)](#preference-alignment-rlhfdpo)
10. [评估 LLM 系统](#evaluation-of-llm-system)
11. [幻觉控制技术](#hallucination-control-techniques)
12. [LLM 部署方案](#deployment-of-llm)
13. [智能体系统](#agent-based-system)
14. [提示注入攻防](#prompt-hacking)
15. [综合话题](#miscellaneous)
16. [实战案例](#case-studies)

---

### 提示工程与 LLM 基础

<details>
<summary>查看问题</summary>

- **生成式 AI 与判别式 AI 的核心区别？**
- **语言模型的训练流程解析**
- **Temperature 参数的作用与设置原则**
- **LLM 解码策略比较分析**
- **如何定义大语言模型的停止条件？**
- **停止序列在 LLM 中的应用方法**
- **提示工程的基本结构是什么？**
- **上下文学习机制解析**
- **提示工程的类型与实施方法**
- **少样本提示的关键注意事项**
- **编写高质量提示的有效策略**
- **如何通过提示工程控制 LLM 幻觉**
- **使用提示工程增强 LLM 推理能力**
- **当思维链(CoT)提示失效时的改进方法**

[查看完整文档 →](/docs/prompt_engineering_basics)

</details>

### 检索增强生成 (RAG)

<details>
<summary>查看问题</summary>

- **如何提高 LLM 输出的准确性和可靠性**
- **RAG 工作机制详细解析**
- **使用 RAG 系统的主要优势**
- **微调 vs RAG 的选择标准**
- **私有数据定制化 LLM 的架构模式**

[查看完整文档 →](/docs/rag_systems)

</details>

### 文档数字化与分块

<details>
<summary>查看问题</summary>

- **分块的基本概念与必要性**
- **影响分块大小的关键因素**
- **不同类型的分块方法比较**
- **寻找最佳分块大小的策略**
- **复杂文档（年报）的数字化处理方案**
- **表格处理的最佳实践**
- **大型表格的检索优化方法**
- **列表项的分块处理技术**
- **生产级文档处理流水线构建**
- **RAG 系统中的图表处理方案**

[查看完整文档 →](/docs/document_processing)

</details>

### 嵌入模型

<details>
<summary>查看问题</summary>

- **向量嵌入与嵌入模型的基本概念**
- **LLM 应用中嵌入模型的使用场景**
- **长短内容嵌入的区别与优化**
- **如何基于私有数据评测嵌入模型**
- **OpenAI 嵌入模型精度不足的优化方案**
- **改进 Sentence Transformer 模型的步骤**

[查看完整文档 →](/docs/embedding_models)

</details>

### 向量数据库原理

<details>
<summary>查看问题</summary>

- **向量数据库的基本原理**
- **向量数据库与传统数据库的差异**
- **索引、数据库与插件的区别**
- **高精度搜索场景下的策略选择**
- **聚类与局部敏感哈希等搜索策略**
- **聚类减少搜索空间的机制**
- **随机投影索引工作原理**
- **局部敏感哈希(LSH)实现机制**
- **乘积量化(PQ)索引方法**
- **不同向量索引的场景应用比较**
- **相似度度量的选择标准**
- **向量数据库过滤的挑战与解决方案**
- **向量数据库选型指南**

[查看完整文档 →](/docs/vector_databases)

</details>

### 高级搜索算法

<details>
<summary>查看问题</summary>

- **信息检索与语义搜索的架构模式**
- **高质量搜索系统的重要性**
- **大规模数据集的高效精准搜索**
- **改进不准确 RAG 检索系统的步骤**
- **基于关键词的检索方法**
- **优化重排模型的微调技术**
- **信息检索常用指标及局限性**
- **类 Quora 系统的评价指标选择**
- **推荐系统的评价指标**
- **不同信息检索指标的应用场景**
- **混合搜索的工作原理**
- **多源搜索结果的合并策略**
- **多轮查询的处理技术**
- **改进检索效果的高级技术**

[查看完整文档 →](/docs/search_algorithms)

</details>

### 语言模型原理

<details>
<summary>查看问题</summary>

- **自注意力机制的详细解析**
- **自注意力的缺陷与改进方案**
- **位置编码的工作原理**
- **Transformer 架构深度剖析**
- **Transformer 相对 LSTM 的优势**
- **局部注意力和全局注意力的区别**
- **Transformer 计算资源消耗优化**
- **扩展 LLM 上下文长度的技术**
- **大词表下的架构优化方案**
- **词表大小平衡策略**
- **不同 LLM 架构的适用场景**

[查看完整文档 →](/docs/llm_internals)

</details>

### 监督微调 (SFT)

<details>
<summary>查看问题</summary>

- **微调的概念与必要性**
- **需要微调的场景分析**
- **微调决策的评估流程**
- **基于上下文的精确回答优化**
- **QA 微调数据集构建方法**
- **微调超参数设置指南**
- **微调基础设施需求估算**
- **消费级硬件的微调方案**
- **参数高效微调(PEFT)方法分类**
- **灾难性遗忘问题解析**
- **重参数化微调方法**

[查看完整文档 →](/docs/fine_tuning)

</details>

### 偏好对齐 (RLHF/DPO)

<details>
<summary>查看问题</summary>

- **选择偏好对齐方法的时机**
- **RLHF 的工作机制与应用**
- **RLHF 中的奖励黑客问题**
- **不同偏好对齐方法比较**

[查看完整文档 →](/docs/preference_alignment)

</details>

### 评估 LLM 系统

<details>
<summary>查看问题</summary>

- **如何评估最适合的 LLM 模型**
- **RAG 系统评估方法论**
- **LLM 评估指标大全**
- **验证链(Chain of Verification)解析**

[查看完整文档 →](/docs/llm_evaluation)

</details>

### 幻觉控制技术

<details>
<summary>查看问题</summary>

- **不同形式的幻觉分类**
- **多层次幻觉控制技术**

[查看完整文档 →](/docs/hallucination_control)

</details>

### LLM 部署方案

<details>
<summary>查看问题</summary>

- **量化不影响精度的原理**
- **LLM 推理吞吐量优化技术**
- **无注意力近似的响应加速方案**

[查看完整文档 →](/docs/llm_deployment)

</details>

### 智能体系统

<details>
<summary>查看问题</summary>

- **智能体基本概念与实现策略**
- **智能体的需求与常见架构**
- **ReAct 提示实现示例**
- **计划与执行策略详解**
- **OpenAI 函数使用实例**
- **OpenAI 函数 vs LangChain 智能体**

[查看完整文档 →](/docs/agent_systems)

</details>

### 提示注入攻防

<details>
<summary>查看问题</summary>

- **提示攻击的基本概念与危害**
- **不同类型提示攻击分析**
- **防御提示攻击的策略**

[查看完整文档 →](/docs/prompt_hacking)

</details>

### 综合话题

<details>
<summary>查看问题</summary>

- **LLM 系统成本优化方案**
- **专家混合模型(MoE)解析**
- **生产级 RAG 系统构建指南**
- **FP8 变量及其优势**
- **无损精度低精度训练技术**
- **KV 缓存大小计算方法**
- **多头注意力层维度分析**
- **注意力层焦点控制技术**

[查看完整文档 →](/docs/miscellaneous)

</details>

### 实战案例

<details>
<summary>查看问题</summary>
敬请期待
</details>

### 🤝 如何贡献

1. 提交新问题到对应分类的 `.md` 文件
2. 完善现有问题答案（需标注引用来源）
3. 改进文档结构或翻译
4. 欢迎提交真实面试经历！

</div>
<div style="grid-column: 2;">


<!-- 英文部分 -->

# LLM Interview All You Need

📚 Curating top company LLM interview questions with community collaboration!
🚀 Continuously updating real questions from Google/OpenAI/Meta/Anthropic covering RAG, fine-tuning, deployment and more.

## Table of Contents

1. [Prompt Engineering & LLM Basics](#prompt-engineering--basics-of-llm)
2. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
3. [Document Digitization & Chunking](#document-digitization--chunking)
4. [Embedding Models](#embedding-models)
5. [Vector DB Internals](#internal-working-of-vector-databases)
6. [Advanced Search Algorithms](#advanced-search-algorithms)
7. [Language Model Internals](#language-models-internal-working)
8. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-of-llm)
9. [Preference Alignment (RLHF/DPO)](#preference-alignment-rlhfdpo)
10. [Evaluating LLM Systems](#evaluation-of-llm-system)
11. [Hallucination Control](#hallucination-control-techniques)
12. [LLM Deployment Strategies](#deployment-of-llm)
13. [Agent-Based Systems](#agent-based-system)
14. [Prompt Hacking Defenses](#prompt-hacking)
15. [Miscellaneous Topics](#miscellaneous)
16. [Case Studies](#case-studies)

---

### Prompt Engineering & Basics

<details>
<summary>View Questions</summary>

- **What is the difference between Predictive/Discriminative AI and Generative AI?**
- **What is LLM, and how are LLMs trained?**
- **What is a token in the language model?**
- **How to estimate the cost of running SaaS-based and Open Source LLM models?**
- **Explain the Temperature parameter and how to set it.**
- **What are different decoding strategies for picking output tokens?**
- **What are different ways you can define stopping criteria in large language model?**
- **How to use stop sequences in LLMs?**
- **Explain the basic structure prompt engineering.**
- **Explain in-context learning**
- **Explain type of prompt engineering**
- **What are some of the aspect to keep in mind while using few-shots prompting?**
- **What are certain strategies to write good prompt?**
- **What is hallucination, and how can it be controlled using prompt engineering?**
- **How to improve the reasoning ability of LLM through prompt engineering?**
- **How to improve LLM reasoning if your COT prompt fails?**

[View Full Document →](/docs/prompt_engineering_basics)

</details>

### Retrieval Augmented Generation

<details>
<summary>View Questions</summary>

- **how to increase accuracy, and reliability & make answers verifiable in LLM**
- **How does RAG work?**
- **What are some benefits of using the RAG system?**
- **When should I use Fine-tuning instead of RAG?**
- **What are the architecture patterns for customizing LLM with proprietary data?**

[View Full Document →](/docs/rag_systems)

</details>

### Document Digitization & Chunking

<details>
<summary>View Questions</summary>

- **What is chunking, and why do we chunk our data?**
- **What factors influence chunk size?**
- **What are the different types of chunking methods?**
- **How to find the ideal chunk size?**
- **What is the best method to digitize and chunk complex documents like annual reports?**
- **How to handle tables during chunking?**
- **How do you handle very large table for better retrieval?**
- **How to handle list item during chunking?**
- **How do you build production grade document processing and indexing pipeline?**
- **How to handle graphs & charts in RAG**

[View Full Document →](/docs/document_processing)

</details>

### Embedding Models

<details>
<summary>View Questions</summary>

- **What are vector embeddings, and what is an embedding model?**
- **How is an embedding model used in the context of LLM applications?**
- **What is the difference between embedding short and long content?**
- **How to benchmark embedding models on your data?**
- **Suppose you are working with an open AI embedding model, after benchmarking accuracy is coming low, how would you further improve the accuracy of embedding the search model?**
- **Walk me through steps of improving sentence transformer model used for embedding?**

[View Full Document →](/docs/embedding_models)

</details>

### Vector DB Internals

<details>
<summary>View Questions</summary>

- **What is a vector database?**
- **Vector DB vs traditional DB**
- **Vector index vs DB vs plugins**
- **High-precision search strategy**
- **Clustering & LSH strategies**
- **Clustering mechanism and limits**
- **Random projection index**
- **Locality-sensitive hashing (LSH)**
- **Product quantization (PQ)**
- **Comparing vector indexes**
- **Selecting similarity metrics**
- **Filtering challenges**
- **Choosing vector databases**

[View Full Document →](/docs/vector_databases)

</details>

### Advanced Search Algorithms

<details>
<summary>View Questions</summary>

- **IR & semantic search patterns**
- **Importance of quality search**
- **Efficient large-scale search**
- **Improving inaccurate RAG retrieval**
- **Keyword-based retrieval**
- **Fine-tuning re-ranking models**
- **Common IR metrics & limitations**
- **Metric for Quora-like systems**
- **Recommendation system metrics**
- **IR metric comparison**
- **Hybrid search mechanics**
- **Merging multi-source results**
- **Multi-hop query handling**
- **Techniques to improve retrieval**

[View Full Document →](/docs/search_algorithms)

</details>

### LLM Internals

<details>
<summary>View Questions</summary>

- **Detailed self-attention explanation**
- **Disadvantages of self-attention**
- **Positional encoding explained**
- **Transformer architecture details**
- **Transformer advantages over LSTM**
- **Local vs global attention**
- **Transformer optimization techniques**
- **Extending LLM context length**
- **Optimizing for large vocabularies**
- **Balancing vocabulary size**
- **LLM architectures & best tasks**

[View Full Document →](/docs/llm_internals)

</details>

### Supervised Fine-Tuning

<details>
<summary>View Questions</summary>

- **Fine-tuning concepts & needs**
- **Scenarios requiring fine-tuning**
- **Making fine-tuning decisions**
- **Context-sufficient answering**
- **Creating Q&A datasets**
- **Hyperparameter configuration**
- **Infrastructure estimation**
- **Consumer hardware fine-tuning**
- **PEFT method categories**
- **Catastrophic forgetting**
- **Re-parameterized methods**

[View Full Document →](/docs/fine_tuning)

</details>

### Preference Alignment

<details>
<summary>View Questions</summary>

- **SFT vs preference alignment**
- **RLHF mechanisms & applications**
- **Reward hacking in RLHF**
- **Preference alignment methods**

[View Full Document →](/docs/preference_alignment)

</details>

### Evaluating LLM Systems

<details>
<summary>View Questions</summary>

- **Evaluating best LLM for task**
- **Assessing RAG systems**
- **Metrics for LLM evaluation**
- **Chain of Verification explained**

[View Full Document →](/docs/llm_evaluation)

</details>

### Hallucination Control

<details>
<summary>View Questions</summary>

- **Forms of hallucinations**
- **Controlling at various levels**

[View Full Document →](/docs/hallucination_control)

</details>

### LLM Deployment

<details>
<summary>View Questions</summary>

- **Why quantization preserves accuracy**
- **LLM inference throughput optimization**
- **Accelerating response time**

[View Full Document →](/docs/llm_deployment)

</details>

### Agent-Based Systems

<details>
<summary>View Questions</summary>

- **Agent concepts & strategies**
- **Need for agents & implementation**
- **ReAct prompting example**
- **Plan and Execute strategy**
- **OpenAI functions examples**
- **OpenAI vs LangChain Agents**

[View Full Document →](/docs/agent_systems)

</details>

### Prompt Hacking

<details>
<summary>View Questions</summary>

- **Prompt hacking explained**
- **Types of prompt hacking**
- **Defense tactics**

[View Full Document →](/docs/prompt_hacking)

</details>

### Miscellaneous

<details>
<summary>View Questions</summary>

- **Optimizing LLM system cost**
- **Mixture of Expert (MoE) models**
- **Building production RAG system**
- **FP8 advantages**
- **Low precision training**
- **KV cache size calculation**
- **Attention layer dimensions**
- **Focusing attention layers**

[View Full Document →](/docs/miscellaneous)

</details>

### Case Studies

<details>
<summary>View Questions</summary>

Coming soon

</details>

### 🤝 How to Contribute

1. Add new questions to corresponding `.md` files
2. Improve existing answers (with citations)
3. Enhance documentation structure or translations
4. Share real interview experiences!

</div>
</div>


## 授权协议 (License)

版权所有 (c) 2025 llm_interview_all_you_need

特此授予任何获得本软件副本及相关文档文件（以下简称"软件"）的任何人免费许可，允许其不受限制地处理本软件，包括但不限于使用、复制、修改、合并、发布、分发、再授权及/或销售软件副本的权利，并允许获得软件的人这样做，但须满足以下条件：

上述版权声明和本许可声明应包含在软件的所有副本或实质性部分中。

本软件按"原样"提供，不提供任何明示或暗示的保证，包括但不限于适销性保证、特定用途适用性保证和非侵权保证。在任何情况下，作者或版权所有者均不对因软件或软件使用或其他交易行为而产生的任何索赔、损害或其他责任负责。
