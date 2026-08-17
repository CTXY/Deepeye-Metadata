"""
Context-graph offline memory loader.

Loads offline_memory_augmentation.jsonl (one object per line), indexed by
(database_id, question_id). Populates mapping_linked_tables_and_columns,
database_schema_after_mapping, mapping_hint (from record field mapping_hint), and
mapping_historical_qa (from historical_pairs).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.dataset import DataItem
from app.db_utils import filter_used_database_schema
from app.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MAPPING_DECISIONS_PATH = (
    _PROJECT_ROOT / "workspace" / "memory" / "context_graph_memory" / "offline_memory_augmentation.jsonl"
)

_index_cache: Optional[Dict[Tuple[str, int], dict]] = None
_index_path: Optional[str] = None


def _load_offline_memory_index(path: str) -> Dict[Tuple[str, int], dict]:
    """Load jsonl; each line must be one record with db_id and question_id."""
    global _index_cache, _index_path
    path_str = str(Path(path).resolve())
    if _index_cache is not None and _index_path == path_str:
        return _index_cache
    index: Dict[Tuple[str, int], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "db_id" not in obj or "question_id" not in obj:
                logger.warning(
                    "Skip offline memory line without db_id/question_id (expected context-graph jsonl)"
                )
                continue
            db_id = obj["db_id"]
            qid = int(obj["question_id"])
            index[(str(db_id), qid)] = obj
    _index_cache = index
    _index_path = path_str
    return index


def get_offline_memory_record(
    database_id: str,
    question_id: int,
    path: Optional[str] = None,
) -> Optional[dict]:
    """Return the offline memory record for (database_id, question_id), or None."""
    if path is None or path == "":
        path = str(DEFAULT_MAPPING_DECISIONS_PATH)
    if not Path(path).exists():
        return None
    index = _load_offline_memory_index(path)
    return index.get((str(database_id), int(question_id)))


def relevant_columns_to_linked_schema(
    relevant_columns: List[str],
    database_schema: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Convert list of "table.column" to Dict[table_name, List[column_name]].
    Only includes (table, column) that exist in database_schema["tables"].
    Column name is resolved to the actual key used in schema (case-sensitive match or key match).
    """
    linked: Dict[str, List[str]] = {}
    tables = database_schema.get("tables") or {}
    for tc in relevant_columns:
        if "." not in tc:
            continue
        table_name, column_name = tc.split(".", 1)
        if table_name not in tables:
            continue
        cols = tables[table_name].get("columns") or {}
        if column_name in cols:
            linked.setdefault(table_name, [])
            if column_name not in linked[table_name]:
                linked[table_name].append(column_name)
            continue
        for actual_col_name in cols:
            if actual_col_name == column_name:
                linked.setdefault(table_name, [])
                if actual_col_name not in linked[table_name]:
                    linked[table_name].append(actual_col_name)
                break
            col_info = cols[actual_col_name]
            if isinstance(col_info, dict) and col_info.get("column_name") == column_name:
                linked.setdefault(table_name, [])
                if actual_col_name not in linked[table_name]:
                    linked[table_name].append(actual_col_name)
                break
    return linked


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


_MAPPING_ID_RE = re.compile(r"K:[A-Za-z0-9]+")


def _rules_entries_for_display(rules: dict) -> List[Tuple[str, str]]:
    """
    Stable ordering: bird-style keys first, then any other string rule fields (e.g. disambiguation_rules).
    """
    r = rules or {}
    out: List[Tuple[str, str]] = []
    preferred = ("discriminative_context", "structural_constraint", "logic_diff")
    for key in preferred:
        val = r.get(key)
        if isinstance(val, str) and val.strip():
            out.append((key, val))
    seen = {k for k, _ in out}
    for key in sorted(r.keys()):
        if key in seen:
            continue
        val = r.get(key)
        if isinstance(val, str) and val.strip():
            out.append((key, val))
    return out


def _mapping_ids_from_rules(rules: dict) -> List[str]:
    """Collect K:... node ids referenced in guidance rule strings (in order, deduped)."""
    chunks = [text for _, text in _rules_entries_for_display(rules)]
    if not chunks:
        return []
    return _dedupe_preserve_order(_MAPPING_ID_RE.findall(" ".join(chunks)))


def _guidance_node_index(guidance_nodes: List[Any]) -> Dict[str, dict]:
    by_id: Dict[str, dict] = {}
    for gn in guidance_nodes or []:
        if not isinstance(gn, dict):
            continue
        mid = gn.get("mapping_node_id")
        if isinstance(mid, str) and mid:
            by_id[mid] = gn
    return by_id


def _format_relevant_mapping_lines(node_by_id: Dict[str, dict], ids: List[str]) -> List[str]:
    lines: List[str] = []
    for mid in ids:
        gn = node_by_id.get(mid)
        if not gn:
            lines.append(f"   - **`{mid}`** — *(no entry in guidance_mapping_nodes)*")
            continue
        lines.append(
            f'   - **`{mid}`** — sql_fragments: {gn.get("sql_fragments") or []}; '
            f'implicit_conditions: {gn.get("implicit_conditions") or []}'
        )
    return lines


def _relevant_columns_from_record(record: dict) -> List[str]:
    cols: List[str] = []
    for uc in record.get("user_choices") or []:
        if not isinstance(uc, dict):
            continue
        for c in uc.get("columns") or []:
            cols.append(c)
    return _dedupe_preserve_order(cols)


def format_offline_memory_mapping_hint(record: dict) -> str:
    """
    Build mapping_hint from user_choices (exact JSON fields) + guidance +
    guidance_mapping_nodes. Does not include historical_pairs.
    """
    parts: List[str] = []
    user_choices = record.get("user_choices") or []
    guidance = record.get("guidance") or []
    guidance_nodes = record.get("guidance_mapping_nodes") or []

    if user_choices:
        parts.append(
            "## User selection (MUST follow — highest priority)\n"
            "The user explicitly chose these mappings. `mention_text` is the **target mention in the question** that needs to be mapped.\n"
            "\n"
            "- `columns`: the user-specified columns that should be used to match this mention.\n"
            "- `sql_fragments`: SQL fragments the user believes are relevant when matching this mention. Treat them as "
            "**reference for how the chosen columns are used and what operations are involved** (e.g., aggregates, "
            "functions, predicates, join/correlation patterns). **Do NOT blindly reuse literals/filters/values** from these fragments if they are specific to a different context.\n"
            "- `implicit_conditions` (if present): conditions that might be applied when generating SQL."
        )
        for i, uc in enumerate(user_choices, 1):
            if not isinstance(uc, dict):
                continue
            mention = uc.get("mention_text", "")
            cols = uc.get("columns") or []
            block_lines = [
                f'{i}. **mention_text (target mention in question):** "{mention}"',
                f"   - **columns (required):** {cols}",
            ]
            sql_frags: List[str] = []
            for s in uc.get("sql_fragments") or []:
                sql_frags.append(s)
            implicit_all: List[str] = []
            node_detail_lines: List[str] = []
            for mn in uc.get("mapping_nodes") or []:
                if not isinstance(mn, dict):
                    continue
                mid = mn.get("mapping_node_id", "")
                for s in mn.get("sql_fragments") or []:
                    sql_frags.append(s)
                for ic in mn.get("implicit_conditions") or []:
                    implicit_all.append(ic)
                node_detail_lines.append(
                    f'     - **mapping_node_id** `{mid}`: **sql_fragments**={mn.get("sql_fragments") or []}, '
                    f'**implicit_conditions**={mn.get("implicit_conditions") or []}'
                )
            sql_frags = _dedupe_preserve_order(sql_frags)
            implicit_all = _dedupe_preserve_order(implicit_all)
            if sql_frags:
                block_lines.append(f"   - **sql_fragments:** {sql_frags}")
            if implicit_all:
                block_lines.append(f"   - **implicit_conditions:** {implicit_all}")
            if node_detail_lines:
                block_lines.append("   - **mapping_nodes:**")
                block_lines.extend(node_detail_lines)
            parts.append("\n".join(block_lines))

    if guidance or guidance_nodes:
        parts.append(
            "## Guidance: mapping and field disambiguation\n"
            "For a given **mention in the question**, history may contain **different possible mappings** (different "
            "columns / fragments) that were previously considered for that same mention.\n"
            "\n"
            "Below are historical records and disambiguation guidance. Use them as **reference information** when "
            "generating SQL (you do not need to copy them verbatim). In particular, SQL fragments here are meant as "
            "**usage patterns** for the columns and operations; if they include specific literal filters/values, you "
            "must re-derive the correct values from the current question instead of reusing them.\n"
            "\n"
            "Each guidance item targets one **mention** (and optional **scope**). **Rules** explain how to choose "
            "between alternatives; **Relevant Mapping** lists `guidance_mapping_nodes` that contrast those "
            "alternatives (matched via `K:...` ids cited in Rules, or the full node list when no ids appear)."
        )
        node_by_id = _guidance_node_index(guidance_nodes)
        gidx = 0
        for g in guidance:
            if not isinstance(g, dict):
                continue
            gidx += 1
            scope = g.get("scope", "")
            mt = g.get("mention_text", "")
            rules = g.get("rules") or {}
            block: List[str] = [f"{gidx}. **target mention:** `{mt}`" if mt else f"{gidx}. **target mention:** *(empty)*"]
            if scope:
                block.append(f"   **scope:** `{scope}`")
            block.append("   **Rules:**")
            rule_entries = _rules_entries_for_display(rules)
            if rule_entries:
                for rk, rv in rule_entries:
                    block.append(f"   - **{rk}:** {rv}")
            else:
                block.append("   - *(no rule fields)*")
            ref_ids = _mapping_ids_from_rules(rules)
            block.append("   **Relevant Mapping:**")
            if ref_ids:
                block.extend(_format_relevant_mapping_lines(node_by_id, ref_ids))
            elif guidance_nodes:
                all_ids = _dedupe_preserve_order(
                    [str(gn.get("mapping_node_id")) for gn in guidance_nodes if isinstance(gn, dict) and gn.get("mapping_node_id")]
                )
                block.extend(_format_relevant_mapping_lines(node_by_id, all_ids))
            else:
                block.append("   - *(none)*")
            parts.append("\n".join(block))

        if gidx == 0 and guidance_nodes:
            block2: List[str] = [
                "1. **target mention:** *(no structured guidance entries; contrast nodes only)*",
                "   **Rules:**",
                "   - *(no `guidance` items in json)*",
                "   **Relevant Mapping:**",
            ]
            all_ids = _dedupe_preserve_order(
                [
                    str(gn.get("mapping_node_id"))
                    for gn in guidance_nodes
                    if isinstance(gn, dict) and gn.get("mapping_node_id")
                ]
            )
            block2.extend(_format_relevant_mapping_lines(node_by_id, all_ids))
            parts.append("\n".join(block2))

    return "\n\n".join(parts).strip()


def historical_pairs_to_mapping_qa(historical_pairs: List[dict]) -> List[Dict[str, Any]]:
    """Convert historical_pairs to mapping_historical_qa items (qa_id, question, sql)."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for hp in historical_pairs or []:
        if not isinstance(hp, dict):
            continue
        qid = hp.get("question_id")
        key = str(qid) if qid is not None else None
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "qa_id": key,
                "question": hp.get("question", ""),
                "sql": hp.get("sql", ""),
            }
        )
    return out


def format_mapping_historical_qa(historical_qa: List[Dict[str, Any]]) -> str:
    if not historical_qa:
        return ""
    lines = [
        "## Reference: Historical Question-SQL Pairs",
        "Historical question–SQL pairs from the same database. Use them for "
        "**similar semantic alignment, column usage, joins, and SQL operations**, not as ground truth "
        "for the current question.",
        "",
    ]
    for idx, qa in enumerate(historical_qa, 1):
        question = qa.get("question", "")
        sql = qa.get("sql", "")
        if question or sql:
            lines.append(f"- Example {idx}: Question: {question}")
            lines.append(f"  SQL: {sql}")
            lines.append("")
    return "\n".join(lines).strip()


def _mapping_hint_from_offline_record(record: dict) -> Optional[str]:
    """Use the JSONL record's pre-written mapping_hint string; no runtime formatting."""
    raw = record.get("mapping_hint")
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    return s if s else None


def ensure_mapping_data(data_item: DataItem, path: Optional[str] = None) -> None:
    """
    Populate mapping_* fields from offline context-graph memory jsonl for this question_id
    and database_id.
    """
    if path is None or path == "":
        path = str(DEFAULT_MAPPING_DECISIONS_PATH)
    record = get_offline_memory_record(data_item.database_id, data_item.question_id, path)
    if not record:
        logger.debug(
            "[Mapping] No offline memory record for: question_id=%s, database_id=%s",
            getattr(data_item, "question_id", "?"),
            data_item.database_id,
        )
        return

    relevant_columns = _relevant_columns_from_record(record)
    schema = data_item.database_schema_after_value_retrieval or data_item.database_schema

    if relevant_columns:
        linked = relevant_columns_to_linked_schema(relevant_columns, schema)
        if linked:
            data_item.mapping_linked_tables_and_columns = linked
            data_item.database_schema_after_mapping = filter_used_database_schema(schema, linked)
        else:
            logger.debug(
                "[Mapping] user_choices columns not resolved in schema for question_id=%s",
                data_item.question_id,
            )
            data_item.mapping_linked_tables_and_columns = None
            data_item.database_schema_after_mapping = None
    else:
        data_item.mapping_linked_tables_and_columns = None
        data_item.database_schema_after_mapping = None

    data_item.mapping_hint = _mapping_hint_from_offline_record(record)
    data_item.mapping_historical_qa = historical_pairs_to_mapping_qa(record.get("historical_pairs") or [])

    if not data_item.mapping_hint and not data_item.mapping_historical_qa:
        data_item.mapping_linked_tables_and_columns = None
        data_item.database_schema_after_mapping = None


def get_sql_generation_hint(data_item: DataItem, use_caf_mapping: bool) -> str:
    """
    Return the hint string for SQL generation/revision/selection prompts.
    When use_caf_mapping: return evidence + mapping_hint + mapping_historical_qa (if any); else evidence only.
    """
    base = data_item.evidence or ""
    if not use_caf_mapping:
        return base
    parts = [base]
    mapping_hint = getattr(data_item, "mapping_hint", None)
    if mapping_hint:
        parts.append(mapping_hint)
    historical_qa = getattr(data_item, "mapping_historical_qa", None)
    if historical_qa:
        parts.append(format_mapping_historical_qa(historical_qa))
    if len(parts) == 1 and not parts[0]:
        return ""
    return "\n\n".join(p for p in parts if p).strip()
