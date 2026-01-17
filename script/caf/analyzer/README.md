# 查询差异分析系统 (Query Difference Analysis System)

## 🎯 核心目标

专为LLM设计的SQL查询差异分析系统，专门回答以下问题：

**"为什么这两个SQL查询结果不同？"**

1. `SELECT ... FROM cards WHERE setCode = 'OGW'`  
2. `SELECT ... FROM cards JOIN set_translations ... WHERE set_translations.setCode = 'OGW'`

## 🚀 核心分析能力

### 两大风险探测

1. **Fan-out Risk (膨胀风险)**: `set_translations`里有多少行对应同一个`setCode`？
   - 导致Query 2结果行数多于Query 1
   
2. **Filtering Risk (过滤风险/丢失风险)**: `cards`里有的`setCode`，是否在`set_translations`里完全缺失？  
   - 导致Query 2查不到Query 1能查到的数据

## 📁 新版文件结构

```
script/caf/analyzer/
├── query_difference_analyzer.py          # 🆕 核心分析引擎（3个精炼函数）
├── query_strategy_report_generator.py    # 🆕 查询策略报告生成器
├── llm_query_difference_interface.py     # 🆕 LLM友好接口
├── test_new_analyzer.py                  # 🆕 新版测试脚本
├── temp/                                 # 🆕 报告存储目录
├── universal_field_analyzer.py           # 原通用分析器（兼容保留）
├── structured_report_generator.py        # 原报告生成器（兼容保留）
└── README.md                             # 本文档
```

## 🔧 3个核心函数（新设计）

### 1. `check_unique_constraint(table, column)`

**目的**: 判断该字段是否具有成为主键（PK）或唯一键（UK）的物理潜质

**分析逻辑**:
- 计算 `Count(*)` 和 `Count(DISTINCT column)`
- 如果两者相等，则为 Unique
- 如果不等，计算重复率

**关键指标**:
- `is_unique`: 是否完全唯一
- `null_count`: 空值数量（主键不能有空）  
- `duplication_rate`: 重复数据的比例
- `can_be_join_key`: 能否作为JOIN的"One"端

### 2. `analyze_join_mapping(left_table, left_col, right_table, right_col)`

**目的**: 🌟**这是最关键的函数**。模拟`LEFT JOIN`和`INNER JOIN`的差异，直接回答"Query 1和Query 2为什么不同"

**分析逻辑**:
- **Fan-out Analysis (膨胀分析)**: 对于Left表中的一个值，Right表中平均有几行？最大有几行？
- **Orphan Analysis (孤儿/丢失分析)**: Left表中有多少比例的值，在Right表中根本找不到？

**关键指标**:
- `max_fan_out`: Left表一个值在Right表最多对应几行（例如返回`5`，意味着`cards`里的一行可能在结果中变成5行）
- `match_ratio`: `cards`中的值在`set_translations`中的覆盖率。如果不是100%，说明Inner Join会丢失数据
- `mapping_type`: 推断关系类型 (1:1, 1:N, N:1, N:N)

### 3. `compare_data_completeness(table_a, col_a, table_b, col_b)`

**目的**: 检查信息完整性（针对`WHERE`子句的有效性）

**分析逻辑**: 比如`setCode`在A表不为空，但在B表可能是NULL

**关键指标**:
- `missing_in_b_samples`: A有但B没有的值（会导致Inner Join过滤掉A的行）
- `null_in_b_count`: B表中该字段为Null的行数

## 🎨 新版报告模板

### 查询策略评估报告 (`cards` vs `set_translations`)

#### 1. 核心差异诊断 (Executive Diagnosis)

> **你的问题**: 使用`JOIN`过滤 (Query 2) 与直接单表过滤 (Query 1) 有区别吗？  
> **最终结论**: **存在重大区别** / **结果基本一致**

| 风险维度 | 检测结果 | 影响解释 |
|:---------|:---------|:---------|
| **数据膨胀 (Fan-out)** | ⚠️ **中等风险** (Max: 5, Avg: 1.2) | `set_translations`中同一个`setCode`对应多条记录。**Query 2会导致`cards`的记录被重复显示**（例如一张牌变成5行）|
| **数据丢失 (Filtering)** | ✅ **低风险** (Match: 99.8%) | 几乎所有`cards`中的`setCode`都能在`set_translations`找到。Inner Join不会无故丢数据 |
| **字段唯一性检查** | `cards.setCode`: **非唯一**<br>`set_translations.setCode`: **非唯一** | 两者均不是唯一键，这是典型的**多对多 (N:N)** 关联，极易产生错误的笛卡尔积 |

#### 2. 场景化模拟 (Scenario Simulation)

针对`WHERE setCode = 'OGW'`的具体情况：

- **Query 1 (单表查询)**:
  - **行为**: 仅查看`cards`表
  - **结果预测**: 返回**200**行（假设OGW系列有200张牌）
  - **准确性**: ✅ 准确反映卡牌数量

- **Query 2 (JOIN查询)**:
  - **行为**: 先将`cards`与`set_translations`连接，再过滤
  - **结果预测**: 返回**1000**行 (假设'OGW'在翻译表中有5种语言)
  - **发生的问题**: 每一张'OGW'的卡牌都重复出现了5次
  - **潜在隐患**: 如果'OGW'在`set_translations`中缺失（虽然概率低），结果将为**0**行

#### 3. 字段关系深度图谱

- **Mapping关系**: `Many-to-Many` (N:N)
  - *解释*: 一个系列有几百张卡(`cards`表不唯一)；一个系列有几种语言的翻译(`set_translations`表不唯一)
- **连接建议**: 
  - ❌ **不建议**直接使用`ON cards.setCode = set_translations.setCode`进行统计查询
  - ✅ **建议**如果必须连接，需要保证`set_translations`的唯一性（例如增加`AND language = 'en'`）

#### 4. 开发指导 (Actionable Advice)

根据分析，建议采用以下策略：

1. **如果你只需要`name`和`colors`(均在cards表)**:
   - 👉 **请使用Query 1 (单表查询)**
   - *理由*: 避免JOIN带来的性能开销和数据重复处理逻辑

2. **如果你确实需要`set_translations`中的信息(比如中文系列名)**:
   - 👉 **请使用Query 2，但必须修改JOIN条件**
   - *代码修正*:
   ```sql
   SELECT c.name, c.colors, t.translation
   FROM cards c
   JOIN set_translations t ON c.setCode = t.setCode
   WHERE t.setCode = 'OGW'
     AND t.language = 'zh-CN'  -- 必须加这个！确保1:1关系，防止膨胀
   ```

## 🚀 LLM使用接口

### 核心函数调用

```python
# 1. 检查字段是否适合作为JOIN key
result = check_field_uniqueness(db_path, "cards", "setCode")
print(f"适合作为JOIN key: {result['can_be_join_key']}")

# 2. 分析查询差异（核心功能）
analysis = analyze_query_difference(
    db_path,
    "cards", "setCode",           # Query 1: SELECT ... FROM cards WHERE setCode = 'OGW'
    "set_translations", "setCode"  # Query 2: SELECT ... FROM cards JOIN set_translations WHERE setCode = 'OGW'
)

# 检查关键风险
print(f"膨胀风险: {analysis['join_mapping']['fan_out_risk']}")
print(f"过滤风险: {analysis['join_mapping']['filtering_risk']}")

# 3. 生成决策报告
report = generate_query_strategy_report(analysis, "cards", "set_translations")
print(f"建议策略: {report['report']['actionable_advice']['priority_recommendations'][0]['strategy']}")
```

### 一站式工作流

```python
# 完整工作流：一次调用获取所有分析结果和报告
result = complete_query_difference_workflow(
    db_path,
    "cards", "setCode",
    "set_translations", "setCode",
    "卡牌表", "系列翻译表"
)

# 快速查看结论
print(result['summary']['conclusion'])

# 查看保存的报告（自动保存到temp/目录）
print(f"详细报告: {result['saved_files']['markdown']}")
```

## 📊 报告存储

- **存储位置**: `/home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyzer/temp/`
- **格式支持**: JSON、Markdown、文本摘要
- **自动命名**: `query_strategy_report_YYYYMMDD_HHMMSS.ext`

## 🧪 测试运行

```bash
cd /home/yangchenyu/DeepEye-SQL-Metadata
python script/caf/analyzer/test_new_analyzer.py
```

测试将演示：
- ✅ 字段唯一性检查
- ✅ 查询差异分析
- ✅ 风险评估 (膨胀 + 过滤)
- ✅ 策略报告生成
- ✅ 完整工作流

## 🔄 与旧版本兼容性

- ✅ 保留原有的`universal_field_analyzer.py`和相关文件
- ✅ 新接口独立运行，不影响现有功能
- ✅ 可以通过`llm_query_difference_interface.py`使用新功能
- ✅ 报告存储位置已更新到`temp/`目录

## 💡 核心改进总结

### 1. 更清晰的术语表达

- **原来**: "JOIN基数" → **现在**: "是否适合作为JOIN key"
- **原来**: "数据膨胀风险" → **现在**: "Fan-out Risk (膨胀风险)"
- **原来**: "最大扇出" → **现在**: "最大扇出: Left表一个值在Right表最多对应几行"

### 2. 专注核心问题

- 直接回答"Query 1 vs Query 2为什么不同"
- 提供具体的SQL修正建议
- 场景化的风险解释

### 3. 改进的报告结构

- 核心差异诊断（风险矩阵）
- 场景化模拟（具体示例）
- 字段关系图谱（连接建议）
- 开发指导（具体SQL代码）

### 4. 更好的存储管理

- 统一存储到`temp/`目录
- 支持多种格式导出
- 自动时间戳命名

---

**Generated for DeepEye-SQL-Metadata project - 专注解决SQL查询差异分析问题**