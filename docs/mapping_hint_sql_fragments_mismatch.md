# Bug: mapping_hint 中 sql_fragments 与 columns 不一致导致 SQL 生成偏差

**发现时间**: 2026-04-04  
**影响范围**: card_games 数据库，qid=439 起的大量问题  
**严重程度**: 高（直接导致 EA 下降，从 ~0.80 跌至 ~0.67）

---

## 问题描述

在 context graph memory augmentation 中，`mapping_hint` 的 `sql_fragments` 字段来自 mapping node（`K:xxxxxx`），这些 fragments 是从历史 SQL 中提取的、与该 mapping node 相关的 SQL 片段。

**核心 bug**：某些 mapping node 的 `sql_fragments` 指向的是**与 `columns` 不同的表/列**，导致 LLM 在生成 SQL 时被误导，选择了错误的列或表。

具体表现为：
- `columns` 字段指定了正确的目标列（如 `sets.code`）
- 但 `sql_fragments` 全部来自另一张表的同义列（如 `set_translations.setcode`）
- LLM 遵循 `sql_fragments` 的模式，生成了使用错误列的 SQL
- 多数投票（consistency scoring）选中了错误的 SQL

---

## 问题根源

mapping node 的 `sql_fragments` 是在 **context graph 构建阶段**从历史 SQL 中挖掘出来的。当两个语义相近的列（如 `sets.code` 和 `set_translations.setcode`）被归入同一个 mapping node，或者 mapping node 的 sql_fragments 来自错误的历史 SQL 上下文时，就会出现 `columns` 与 `sql_fragments` 指向不同表的情况。

**相关代码位置**：
- mapping hint 格式化：`app/prompt/factory.py` → `get_sql_generation_hint()`
- context graph augmentation：`app/pipeline/memory_augmentation/context_graph.py` → `augment()`
- mapping node 构建：`workspace/memory/context_graph_memory/offline_memory_augmentation.jsonl`

---

## 典型案例

### Case 1: qid=439 — `sets.code` vs `set_translations.setcode`

**问题**: `List out the set name of the set code "ALL".`  
**Gold SQL**: `SELECT name FROM sets WHERE code = 'ALL'`  
**Gold 结果**: `[('Alliances',)]`

**Memory Augmentation 给出的 mapping_hint**:
```
mention_text: "set code ALL"
columns (required): ['sets.code']          ← 正确：指向 sets 表
sql_fragments: [
  '"set_translations"."setcode"',          ← 错误：全部来自 set_translations 表
  '"set_translations"."setcode" = '10E'',
  '"set_translations"."setcode" = 'ARC''
]
mapping_node: K:000060
```

**影响**：LLM 被 sql_fragments 引导，大量生成 `SELECT code FROM sets WHERE code = 'ALL'`（返回 `'ALL'`）而非 `SELECT name FROM sets WHERE code = 'ALL'`（返回 `'Alliances'`）。

**24 个候选的执行结果分布**：
| 结果 | 票数 | 是否正确 |
|------|------|----------|
| `('ALL',)` — SELECT code | 11 | ❌ 错误 |
| `('Alliances',)` — SELECT name | 9 | ✅ 正确 |
| `(19,)` — SELECT id | 1 | ❌ 错误 |
| JOIN 查询报错（T2.name 不存在） | 3 | ❌ 错误 |

多数投票选中了错误答案 `code='ALL'`，EA=0。

---

### Case 2: qid=441 — `sets.code` 被误导为不需要 JOIN

**问题**: `State the set code of the set with release date of 07/13/2007?`  
**Gold SQL**: `SELECT T2.setCode FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.releaseDate = '2007-07-13'`  
**Gold 结果**: `[('10E',), ('10E',), ...]`（10 行，通过 set_translations JOIN 展开）

**Memory Augmentation 给出的 mapping_hint**:
```
mention_text: "set code"
columns (required): ['sets.code']
sql_fragments: [
  '"set_translations"."setcode"',
  '"set_translations"."setcode" = '10E'',
  '"set_translations"."setcode" = 'ARC''
]
mapping_node: K:000060
```

**影响**：LLM 生成了 `SELECT code FROM sets WHERE releaseDate = '2007-07-13'`（返回 `[('10E',), ('P10E',)]`，2 行），而 gold 要求通过 `set_translations` JOIN 返回 10 行。

**24 个候选的执行结果分布**：
| 结果 | 票数 | 是否正确 |
|------|------|----------|
| `{('10E',), ('P10E',)}` — 直接查 sets | 23 | ❌ 错误（缺少 JOIN） |
| `{('10E',)}` — 其他变体 | 1 | ❌ 错误 |

EA=0，且只有 1 个唯一结果集，触发 "Only one valid SQL candidate, directly select it"。

---

### Case 3: qid=443 — `sets.code` 的 sql_fragments 来自 `sets.type`

**问题**: `Give the code of sets have expansion type of 'expansion'?`  
**Gold SQL**: `SELECT code FROM sets WHERE type = 'expansion'`

**Memory Augmentation 给出的 mapping_hint**:
```
mention_text: "set codes"
columns (required): ['sets.code']
sql_fragments: [
  '"sets"."type"',
  '"sets"."type" = 'expansion''
]
mapping_node: K:000083
```

这里 `columns` 是 `sets.code`，但 `sql_fragments` 是 `sets.type` 的片段。虽然这个案例 LLM 最终生成了正确的 SQL（因为 `type='expansion'` 是 WHERE 条件，不影响 SELECT），但 mapping_hint 的语义是混乱的——它把 SELECT 目标列（`code`）和 WHERE 条件列（`type`）混在了同一个 mapping node 里。

---

### Case 4: qid=444 — `foreign name` 被映射到 `cards.name` 而非 `foreign_data.name`

**问题**: `Name the foreign name of the card that has boros watermark? List out the type of this card.`  
**Gold SQL**: `SELECT DISTINCT T1.name, T1.type FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.watermark = 'boros'`

**Memory Augmentation 给出的 mapping_hint**:
```
mention_text: "foreign name of card"
columns (required): ['cards.name']         ← 错误：应为 foreign_data.name
sql_fragments: ['"cards"."name"']
```

**影响**：`foreign name` 语义上应该指 `foreign_data.name`（外语名称），但 mapping 把它指向了 `cards.name`（英文名称）。LLM 生成的 SQL 选择了 `cards.name` 而非 `foreign_data.name`，导致返回英文名而非外语名。

**24 个候选的执行结果分布**：
| 结果类型 | 票数 | 是否正确 |
|----------|------|----------|
| `(cards.name, None)` — 缺少 foreign_data.name | 17 | ❌ 错误 |
| `(foreign_data.name,)` — 只有外语名 | 3 | ❌ 错误（缺 type） |
| `(cards.name, cards.type)` — 正确列但错误表 | 2 | ❌ 错误 |
| 其他 | 2 | ❌ 错误 |

---

## 问题模式总结

| 模式 | 描述 | 案例 |
|------|------|------|
| **跨表同义列混淆** | `columns` 指向表 A 的列，`sql_fragments` 来自表 B 的同义列 | qid=439,441,442: `sets.code` vs `set_translations.setcode` |
| **SELECT 列与 WHERE 列混淆** | `columns` 是 SELECT 目标，`sql_fragments` 是 WHERE 条件列 | qid=443: `sets.code` vs `sets.type` |
| **语义歧义列错误映射** | 自然语言中的 "foreign name" 被映射到错误的物理列 | qid=444: `foreign name` → `cards.name` 而非 `foreign_data.name` |

---

## 修复方向

### 方向 1：修复 mapping node 的 sql_fragments 生成逻辑

**位置**: context graph 构建阶段（`offline_memory_augmentation.jsonl` 的生成代码）

问题：mapping node 的 `sql_fragments` 应该只包含与该 node 对应的 `columns` 直接相关的 SQL 片段，而不是来自语义相近但不同表的列。

修复思路：
- 在构建 mapping node 时，过滤掉 `sql_fragments` 中与 `columns` 不属于同一张表的片段
- 或者在 `augment()` 中，在格式化 mapping_hint 之前，对 `sql_fragments` 做一次列归属校验

### 方向 2：在 prompt 中降低 sql_fragments 的权重

**位置**: `app/prompt/factory.py` → `get_sql_generation_hint()` 或 `app/prompt/prompt_template.py`

当前 prompt 中 mapping_hint 的措辞是 "MUST follow — highest priority"，这使得 LLM 过度依赖 `sql_fragments`。可以：
- 将 `sql_fragments` 的说明改为 "参考用途，不强制遵循"
- 或者在 `columns` 与 `sql_fragments` 所属表不一致时，不输出 `sql_fragments`

### 方向 3：在 `_get_top_k_sql_candidates` 中引入 gold schema 约束

**位置**: `app/pipeline/sql_selection/br_selection.py` → `_get_top_k_sql_candidates()`

在候选 SQL 过滤时，可以检查候选 SQL 是否使用了 mapping_hint 中 `columns` 指定的列，优先保留与 `columns` 一致的候选。

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `workspace/memory/context_graph_memory/offline_memory_augmentation.jsonl` | mapping node 数据，包含有问题的 K:000060 等节点 |
| `app/pipeline/memory_augmentation/context_graph.py` | augment() 方法，格式化 mapping_hint |
| `app/prompt/factory.py` | get_sql_generation_hint()，控制 hint 的格式和权重 |
| `app/prompt/prompt_template.py` | mapping_hint 的 prompt 模板文字 |
| `app/pipeline/sql_selection/br_selection.py` | _get_top_k_sql_candidates()，候选过滤逻辑 |

---

## 诊断方法

快速定位某个 qid 是否受此问题影响：

```bash
# 1. 查看该 qid 的 mapping_hint
grep -A 30 "question_id=<QID>" workspace/logs/<log_file>.log

# 2. 检查 columns 与 sql_fragments 是否属于同一张表
# 如果 columns=['sets.code'] 但 sql_fragments 包含 'set_translations.setcode'，则受影响

# 3. 查看候选 SQL 分布
python3 -c "
import pickle, sqlite3
db = 'data/bird/dev/dev_databases/<db_id>/<db_id>.sqlite'
conn = sqlite3.connect(db)
with open('workspace/sql_generation/bird/<dataset>.pkl', 'rb') as f:
    import pickle; dataset = pickle.load(f)
for item in dataset:
    if int(item.question_id) == <QID>:
        from collections import Counter
        results = []
        for sql in item.sql_candidates:
            try: results.append(frozenset(conn.execute(sql).fetchall()))
            except: results.append('ERROR')
        for r, c in Counter(results).most_common(): print(c, r)
"
```
