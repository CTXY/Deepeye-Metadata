# 分析：Context Graph Memory Augmentation 在 card_games 数据库上的性能退化

**分析时间**: 2026-04-05  
**数据集**: BIRD-dev，card_games 数据库，qid 439–449  
**对比基准**: 无 memory augmentation（`sub_dev_school_and_card_latter50.pkl`）  
**加入 memory 后**: context graph 策略（`school_and_card_latter50_context_graph.pkl`）  
**结论**: memory augmentation 在该区间引入了新的错误，且完全正确的 qid=439 也因此答错

---

## 总体对比

| qid | 问题摘要 | 无 memory（正确？） | 有 memory（正确？） |
|-----|----------|---------------------|---------------------|
| 439 | set name of code "ALL" | ✅ | ❌ |
| 440 | foreign language of "A Pedra Fellwar" | ❌ | ❌ |
| 441 | set code with release date 07/13/2007 | ❌ | ❌ |
| 442 | base set size and set code in block Masques/Mirage | ❌ | ❌ |
| 443 | code of sets with expansion type | ❌ | ❌ |
| 444 | foreign name + type of card with boros watermark | ❌ | ❌ |
| 445 | language + flavor text of card with colorpie watermark | ❌ | ❌ |
| 446 | % of cards with convertedManaCost=10 in set of Abyssal Horror | ❌ | ❌ |
| 447 | code of sets with expansion commander type | ❌ | ❌ |
| 448 | foreign name + type of card with abzan watermark | ❌ | ❌ |
| 449 | language + type of card with azorius watermark | ❌ | ❌ |

qid=439 是唯一由 memory augmentation 直接导致退化的案例（从正确变错误）。其余案例无 memory 时也已答错，但 memory 注入了错误的约束，使得正确答案在候选中的比例进一步降低。

---

## 三类根因

### 类型 A：mapping hint 中 columns 的语义映射错误

**受影响 QID**：444、448

这两题问的是 "foreign name of card"（外语名称），Gold SQL 里对应的是 `foreign_data.name`（通过 JOIN foreign_data 获取）。但 mapping hint 注入了：

```
mention_text: "foreign name of card with boros watermark"
columns (required): ['cards.name']
sql_fragments: ['"cards"."name"']
```

`cards.name` 是英文名，不是外语名。Gold SQL 中确实包含 `T1.name`（来自 cards 表），但那是作为 SELECT 目标之一被 JOIN 携带出来的英文名，而非 "foreign name" 的来源列。评估机制只检查了 column 是否在 Gold SQL 中出现，没有验证其**语义角色**，导致这个错误映射通过了评估。

LLM 被 `required` 约束强制使用 `cards.name`，生成了：

```sql
-- qid=444，LLM 生成（错误）
SELECT "cards"."name", "cards"."originaltype"
FROM "cards"
WHERE LOWER("cards"."name") LIKE '%boros%'
```

Gold SQL：
```sql
SELECT DISTINCT T1.name, T1.type
FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid
WHERE T1.watermark = 'boros'
```

错误链：① `cards.name` 被 required → 不需要 JOIN foreign_data → `watermark` 过滤也丢失，改成了 LIKE name 模糊匹配。

---

### 类型 B：sql_fragments 包含大量同列枚举值，干扰 LLM 的过滤判断

**受影响 QID**：440、445、449（均来自同一 mapping node `K:000026`）

这三个问题的 mention 均被映射到 `foreign_data.language`，且注入了以下 sql_fragments：

```
"foreign_data"."language" = 'Chinese Simplified'
"foreign_data"."language" = 'French'
"foreign_data"."language" = 'German'
"foreign_data"."language" = 'Japanese'
"foreign_data"."language" = 'Phyrexian'
"foreign_data"."language" = 'Russian'
"foreign_data"."language" = 'Spanish'
```

这些 fragments 是历史上不同问题中对该列使用过的具体过滤值。它们对当前问题毫无意义（当前问题只需要 SELECT language，不需要过滤 language），却以 `sql_fragments` 的形式注入了 prompt。

qid=440 的 Gold SQL：
```sql
SELECT DISTINCT language FROM foreign_data WHERE name = 'A Pedra Fellwar'
```

LLM 生成（错误）：
```sql
SELECT "foreign_data"."language" FROM "foreign_data"
```

WHERE 条件（`name = 'A Pedra Fellwar'`）完全消失。推断原因：7 个语言枚举 fragment 互相矛盾，LLM 无法选择，放弃了 WHERE 过滤；与此同时，过多无关 fragment 转移了 LLM 对真正过滤条件（按卡名过滤）的注意力。

---

### 类型 C：mention 边界定义有误 / required columns 阻断了正确 JOIN 路径

**受影响 QID**：439、441、442、443、447

这类问题的共同特征：Gold SQL 要求 `sets JOIN set_translations`，从 `set_translations.setCode` 获取 set code。但 mapping hint 注入了：

```
mention_text: "set code" / "set codes"
columns (required): ['sets.code']
sql_fragments: ['"sets"."code"']
```

`sets.code` 和 `set_translations.setCode` 在语义上等价（都是 set 的代码），但它们在 Gold SQL 的设计中承担不同角色：`sets.code` 是 JOIN key，`set_translations.setCode` 是 SELECT 目标，通过 JOIN 展开成多行（每个翻译语言一行）。mapping 把 "set code" 固定到 `sets.code`，LLM 就不再生成 JOIN，直接查 sets 表，导致行数少了一个数量级。

**qid=439 的特殊失败**（baseline 正确，memory 后答错）：

问题：`List out the set name of the set code "ALL".`  
Gold：`SELECT name FROM sets WHERE code = 'ALL'`

mapping hint 把 mention "sets with code ALL" 映射到 `sets.code` 并注入 fragment `"sets"."code" = 'ALL'`。该 mention 语义上是**过滤条件**（WHERE code = 'ALL'），而非 SELECT 目标。但 LLM 收到 `columns: ['sets.code']` 的 required 约束后，在 SELECT 中输出了 code 而非 name：

```sql
-- LLM 生成（错误）
SELECT "code" FROM "sets" WHERE "code" = 'ALL'
-- 返回 ('ALL',)，而非 ('Alliances',)
```

根本问题：mention "sets with code ALL" 的**语义角色**是 filter，mapping 的 `columns` 字段不应该把 filter 列标记为 required SELECT 列。

---

## 评估机制的盲区

context graph 构建阶段对 mapping 进行了多维度评估，但存在以下盲区：

1. **只检查 column 是否在 Gold SQL 中出现，未检查语义角色**  
   `cards.name` 出现在 Gold SQL 的 SELECT 中，但在该 qid 的上下文里它是英文名，而 mention 描述的是外语名。评估通过了，但语义是错的（A 类问题）。

2. **未检查 sql_fragments 对当前问题是否有意义**  
   大量历史枚举值 fragments 通过了评估（因为它们确实来自该列的历史使用），但对当前问题它们是噪声（B 类问题）。

3. **未区分 mention 的语义角色（SELECT target vs. filter/join key）**  
   过滤条件 mention 被映射的 columns 成了 required，导致 LLM 把它放入 SELECT（C 类问题）。

---

## 修复建议

### 针对 B 类（优先级高，实现最简单）

当一个 mapping node 的 sql_fragments **全部都是同一列的枚举过滤**（`col = 'v1'`, `col = 'v2'`...），不应该注入这些 fragments，只保留列名。

判断条件：fragments 去除列名后，剩余部分全为 `= 'xxx'` 形式 → 视为枚举过滤，丢弃。

修改位置：`app/pipeline/memory_augmentation/context_graph.py` 中格式化 mapping_hint 的部分。

### 针对 C 类

`columns` 字段应列出**所有等价路径**，例如同时包含 `['sets.code', 'set_translations.setCode']`，而非只选一个。这样 LLM 可以根据查询需求选择正确的路径。

或者：降低 `required` 的强制程度，改为 `preferred`，允许 LLM 根据 JOIN 需求做选择。

修改位置：context graph 构建阶段（`offline_memory_augmentation.jsonl` 的生成逻辑），或 `app/prompt/prompt_template.py` 中 mapping_hint 的措辞。

### 针对 A 类（根治）

评估机制需要增加**语义角色校验**：当 mention 描述的是 "foreign name" 这类带有来源表语义的表达时，不能仅靠 column 是否在 Gold SQL 中出现来判断映射正确性，还需要检查该 column 在 Gold SQL 中的角色（SELECT 目标 / JOIN key / filter）是否与 mention 的语义一致。

短期缓解方案：对包含 "foreign" 关键词的 mention，优先检查 `foreign_data` 表的相关列，而非直接接受 `cards` 表的列。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `workspace/logs/20260404172338.log` | 本次运行的 memory augmentation 日志，包含各 qid 的 mapping_hint 内容 |
| `workspace/sql_selection/bird/school_and_card_latter50_context_graph.pkl` | 有 memory 的最终结果 pkl |
| `workspace/sql_selection/bird/sub_dev_school_and_card_latter50.pkl` | 无 memory 的 baseline 结果 pkl |
| `workspace/memory/context_graph_memory/offline_memory_augmentation.jsonl` | context graph mapping 数据源 |
| `app/pipeline/memory_augmentation/context_graph.py` | mapping_hint 格式化逻辑 |
| `app/prompt/factory.py` | `get_sql_generation_hint()`，控制 hint 的格式和权重 |
| `docs/mapping_hint_sql_fragments_mismatch.md` | 早期相关 bug 记录（主要记录 K:000060 节点的跨表列混淆问题） |
