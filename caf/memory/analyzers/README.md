# Ambiguous Column Analyzers

模糊字段分析器 - 实现 Ambiguity Discriminator Matrix，帮助 Text-to-SQL 系统正确区分语义相似的字段。

## 概述

当数据库中存在多个语义相似或值重叠的字段时（例如 `order_date` vs `shipping_date`），Text-to-SQL 系统容易混淆。本模块通过两个维度深度分析这些模糊字段对，生成详细的差异档案（Diff Profile），帮助 Agent 做出正确选择。

## 架构设计

```
Mining Stage (挖掘)
├── PseudoQueryCollisionMiner  → 语义相似字段对
├── ValueOverlapClusterMiner   → 值重叠字段对
└── Output: AmbiguousPair (存储实际冲突的 pairs)

Analysis Stage (分析)
├── DataContentAnalyzer         → 数据内容维度
│   ├── 集合拓扑分析 (Set Topology)
│   ├── 逻辑约束检测 (Logical Constraints)
│   └── 结果敏感性分析 (Result Sensitivity)
├── SemanticIntentAnalyzer      → 语义意图维度 (LLM)
│   ├── 实体/属性归属 (Entity Alignment)
│   └── 判别性场景生成 (Scenario Generation)
└── Output: DiffProfile (完整的差异档案)

Storage
└── AmbiguousPairStore (Parquet files per database)
```

## 数据结构

### AmbiguousPair

存储一对模糊字段及其差异分析：

```python
AmbiguousPair(
    pair_id="db_001_pair_0001",
    database_id="california_schools",
    column_a=DBElementRef(table_name="schools", column_name="city"),
    column_b=DBElementRef(table_name="schools", column_name="district"),
    discovery_methods=["pseudo_query_collision", "value_overlap"],
    semantic_collision_score=0.85,
    value_jaccard=0.65,
    diff_profile=DiffProfile(...)
)
```

### DiffProfile

综合两个维度的分析结果：

```python
DiffProfile(
    data_content_profile=DataContentProfile(
        set_relationship="overlapping",
        jaccard_similarity=0.65,
        constraint_rule="A != B",
        sensitivity_type="high_sensitivity"
    ),
    semantic_intent_profile=SemanticIntentProfile(
        semantic_nuance="City is a geographic location; District is an administrative region",
        scenario_a="Show schools in San Francisco",
        scenario_b="Show schools in Unified School District",
        trigger_keywords_a=["city", "location", "where"],
        trigger_keywords_b=["district", "region", "administrative"]
    ),
    guidance_rule="Use Column A for geographic queries; Use Column B for administrative boundaries"
)
```

## 使用方法

### 1. 命令行工具

```bash
# 完整分析（挖掘 + 分析）
python script/caf/analyze_ambiguous_pairs.py --database-id california_schools

# 只用 LLM 分析（跳过数据内容分析，更快）
python script/caf/analyze_ambiguous_pairs.py --database-id california_schools --no-data-content

# 使用更多 worker 并行分析
python script/caf/analyze_ambiguous_pairs.py --database-id california_schools --workers 8
```

### 2. Python API

```python
from caf.memory.analyzers import AmbiguityAnalyzer

# 初始化 analyzer
analyzer = AmbiguityAnalyzer(
    pair_store=pair_store,
    pseudo_query_miner=pseudo_query_miner,
    value_overlap_miner=value_overlap_miner,
    data_content_analyzer=data_content_analyzer,
    semantic_intent_analyzer=semantic_intent_analyzer,
    config={
        "num_workers": 4,
        "enable_data_content": True,
        "enable_semantic_intent": True,
    }
)

# 运行分析
stats = analyzer.analyze_database("california_schools")

# 查询结果
pair_store.bind_database("california_schools")
pairs = pair_store.list_pairs()

for pair in pairs:
    if pair.diff_profile:
        print(f"Pair: {pair.column_a.column_name} vs {pair.column_b.column_name}")
        print(f"Guidance: {pair.diff_profile.guidance_rule}")
```

### 3. 查询特定字段的模糊对

```python
# 查询包含特定列的所有模糊对
pairs = pair_store.get_pairs_for_column("schools", "city")

for pair in pairs:
    other_col = pair.column_b if pair.column_a.column_name == "city" else pair.column_a
    print(f"'city' is ambiguous with: {other_col.column_name}")
    
    if pair.diff_profile and pair.diff_profile.semantic_intent_profile:
        print(f"  Difference: {pair.diff_profile.semantic_intent_profile.semantic_nuance}")
```

## 分析维度详解

### 维度一：数据内容维度 (Data Content Perspective)

通过统计学和集合论，量化"错选"的代价和物理规律。

#### 1. 集合拓扑分析 (Set Topology)

分析两列的非空值集合关系：

- **A ⊂ B (子集)**: A 是 B 的特例，B 是通例
- **A ∩ B ≈ ∅ (互斥)**: 两列描述不同的维度
- **重叠 (Overlapping)**: 部分共享值

**示例:**
```
Column A (city): {"San Francisco", "Los Angeles", "Oakland"}
Column B (district): {"SF Unified", "LA Unified", "Oakland Unified"}
→ 互斥关系 (Jaccard < 0.1)
```

#### 2. 逻辑约束检测 (Logical Constraints)

检测隐式的逻辑规则（仅当两列在同一表时）：

- **A ≤ B**: 例如 `order_date ≤ shipping_date`
- **A == B**: 两列实际上是同义词
- **No constraint**: 无明显规则

**示例:**
```
Table: orders
- order_date: 2024-01-01
- shipping_date: 2024-01-05
→ Constraint: order_date ≤ shipping_date (违反率 < 5%)
```

#### 3. 结果敏感性分析 (Result Sensitivity)

模拟：在 WHERE 子句中用错字段会怎样？

- **低敏感性 (Low Sensitivity)**: 结果集高度重叠 → 同义词，安全
- **高敏感性 (High Sensitivity)**: 结果集完全不同 → 危险！
- **上下文依赖 (Context Dependent)**: 部分重叠

**方法:**
1. 找出 A 和 B 的共同值
2. 对每个共同值 v，比较 `SELECT * WHERE A=v` 和 `SELECT * WHERE B=v` 的结果集
3. 计算结果集的 Jaccard 相似度

### 维度二：语义意图维度 (Semantic Intent Perspective)

利用 LLM 的常识推理，挖掘设计者的意图和业务边界。

#### 1. 实体/属性归属 (Entity Alignment)

识别每个字段描述的主体：

**示例:**
```
Column A: order_date
→ Entity: "Order entity - when the customer placed the order"

Column B: shipping_date
→ Entity: "Shipment entity - when the warehouse dispatched the package"
```

#### 2. 判别性场景生成 (Discriminative Scenario Generation)

生成成对的用户问题，明确什么时候用哪个字段：

**示例:**
```
Scenario A (必须用 order_date):
"Show me the total sales for last month"
→ 关键词: ["sales", "bought", "purchased", "ordered"]

Scenario B (必须用 shipping_date):
"How many orders were shipped on time?"
→ 关键词: ["shipped", "delivered", "dispatched", "logistics"]

Discriminative Logic:
"order_date reflects business performance (revenue timing),
 shipping_date reflects operational efficiency (delivery performance)"
```

## 工作流程

### Phase 1: Mining (挖掘)

```
Database → Miners → Raw Pairs
├── PseudoQueryCollisionMiner
│   └── 为每列生成查询 → SPLADE 检索 → 检测冲突
└── ValueOverlapClusterMiner
    └── 提取列值 → 计算 Jaccard → 识别高重叠对
```

### Phase 2: Deduplication (去重)

```
Raw Pairs → Deduplicate → Unique Pairs
- 相同的 (col_a, col_b) 合并
- 合并 discovery_methods: ["pseudo_query_collision", "value_overlap"]
- 保留最高分数: max(semantic_score), max(jaccard)
```

### Phase 3: Analysis (分析)

```
Unique Pairs → Parallel Analysis → Analyzed Pairs
├── Thread 1: analyze_pair_1 (Data Content + Semantic Intent)
├── Thread 2: analyze_pair_2
├── Thread 3: analyze_pair_3
└── Thread N: analyze_pair_N

每个 pair:
1. DataContentAnalyzer → DataContentProfile
2. SemanticIntentAnalyzer (LLM) → SemanticIntentProfile
3. Merge → DiffProfile + GuidanceRule
```

### Phase 4: Storage (存储)

```
Analyzed Pairs → Parquet File
Path: memory/ambiguous_pairs/ambiguous_pairs_{database_id}.parquet

字段:
- pair_id, database_id
- column_a, column_b
- discovery_methods, scores
- diff_profile (JSON)
```

## 性能优化

### 1. 多线程并行

```python
config = {
    "num_workers": 8,  # 根据 CPU 核心数调整
}
```

- LLM 分析：I/O 密集，适合高并发
- 数据内容分析：CPU + I/O 密集，适度并发

### 2. 选择性分析

```python
# 只做 LLM 分析（跳过 SQL 分析，快 5-10 倍）
config = {
    "enable_data_content": False,
    "enable_semantic_intent": True,
}

# 只做数据内容分析（不调用 LLM）
config = {
    "enable_data_content": True,
    "enable_semantic_intent": False,
}
```

### 3. 增量分析

```python
# 只分析未分析的 pairs
pair_store.bind_database(database_id)
pairs = pair_store.list_pairs()

unanalyzed = [p for p in pairs if p.diff_profile is None]
# 只分析 unanalyzed
```

## 配置选项

### DataContentAnalyzer

```python
config = {
    "constraint_sample_size": 10000,  # 约束检测的样本数
    "sensitivity_sample_size": 10,    # 敏感性分析的样本数
    "min_common_values": 5,           # 最少共同值数量
}
```

### SemanticIntentAnalyzer

```python
config = {
    "temperature": 0.3,   # LLM 温度（低温 = 更确定）
    "max_retries": 2,     # 失败重试次数
}
```

### AmbiguityAnalyzer

```python
config = {
    "num_workers": 4,                 # 并行 worker 数量
    "enable_data_content": True,      # 是否启用数据内容分析
    "enable_semantic_intent": True,   # 是否启用语义意图分析
}
```

## 输出示例

### 统计信息

```
================================================================================
ANALYSIS COMPLETE
================================================================================
Database: california_schools
Total pairs: 15
Successfully analyzed: 13
Failed: 2

Discovery methods:
  - pseudo_query_collision: 8
  - value_overlap: 10
  - both: 3
================================================================================
```

### DiffProfile 示例

```json
{
  "data_content_profile": {
    "set_relationship": "overlapping",
    "containment_a_in_b": 0.45,
    "containment_b_in_a": 0.52,
    "jaccard_similarity": 0.38,
    "constraint_rule": null,
    "sensitivity_type": "high_sensitivity",
    "avg_result_overlap": 0.12
  },
  "semantic_intent_profile": {
    "semantic_nuance": "City refers to the geographic municipality where a school is located. District refers to the administrative school district that manages the school, which may span multiple cities.",
    "scenario_a": "Which schools are located in San Francisco?",
    "scenario_b": "Which schools belong to SF Unified School District?",
    "trigger_keywords_a": ["city", "location", "municipality", "where"],
    "trigger_keywords_b": ["district", "school district", "administrative", "managed by"]
  },
  "guidance_rule": "HIGH RISK: Swapping these columns will return completely different results | Use Column A for: [city, location, where]; Use Column B for: [district, school district, administrative]"
}
```

## 集成到 Text-to-SQL Pipeline

### 1. Schema Linking 阶段

```python
# 当检测到多个候选列时，查询 DiffProfile
candidates = ["schools.city", "schools.district"]
pairs = pair_store.get_pair("schools", "city", "schools", "district")

if pairs and pairs.diff_profile:
    # 使用 trigger_keywords 进行匹配
    query = "schools in San Francisco"
    profile = pairs.diff_profile.semantic_intent_profile
    
    if any(kw in query.lower() for kw in profile.trigger_keywords_a):
        selected = "schools.city"
    elif any(kw in query.lower() for kw in profile.trigger_keywords_b):
        selected = "schools.district"
```

### 2. SQL Generation Prompt 增强

```python
# 在 prompt 中注入 guidance_rule
prompt = f"""
Generate SQL for: {user_query}

Schema: {schema}

⚠️ DISAMBIGUATION GUIDANCE:
{pairs.diff_profile.guidance_rule}

Generate SQL:
"""
```

## 故障排查

### 问题 1: 数据库路径未找到

```
ERROR: Database path not found for pair xxx (database_id: yyy)
```

**解决:** 确保 `memory/database_mapping.json` 包含正确的映射：

```json
{
  "/path/to/database.sqlite": "california_schools"
}
```

### 问题 2: LLM 调用失败

```
WARNING: LLM call failed (attempt 1/2): ...
```

**解决:** 检查 LLM 配置（API key, model name, base_url）。

### 问题 3: 分析速度慢

**解决:**
1. 增加 workers: `--workers 8`
2. 跳过数据内容分析: `--no-data-content`
3. 减少敏感性采样: `sensitivity_sample_size=5`

## 未来扩展

### 1. 更多分析维度

- **Frequency Distribution**: 值频率分布差异
- **Temporal Patterns**: 时间序列模式差异
- **Cardinality Ratio**: 基数比率分析

### 2. 主动学习

- 从用户反馈中学习 trigger keywords
- 自动更新 guidance_rules

### 3. 可视化

- 生成字段对的差异可视化图表
- 交互式 DiffProfile 浏览器

## 参考

- **Ambiguity Discriminator Matrix**: 原始设计文档
- **SPLADE**: Sparse Lexical And Expansion Model for retrieval
- **MinHash**: Locality-Sensitive Hashing for Jaccard similarity

---

**维护者**: DeepEye-SQL-Metadata Team  
**最后更新**: 2025-01-16












