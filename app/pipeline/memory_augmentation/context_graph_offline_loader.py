"""
Context-graph offline memory loader (standalone).

This module was extracted from the deprecated mapping-analysis stage so that
memory augmentation can operate without any dependency on
`app.pipeline.mapping_analysis`.
"""

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.dataset import DataItem
from app.db_utils import filter_used_database_schema  # kept for backward compatibility
from app.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MAPPING_DECISIONS_PATH = (
    _PROJECT_ROOT
    / "workspace"
    / "memory"
    / "context_graph_memory"
    / "offline_memory_augmentation.jsonl"
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

    def _norm(s: str) -> str:
        # Offline memory user_choices may use different casing (e.g. setcode vs setCode).
        # Schema keys preserve DB casing, so normalize for robust matching.
        return (s or "").strip().lower()

    linked: Dict[str, List[str]] = {}
    tables = database_schema.get("tables") or {}

    # Build case-insensitive lookup for table names.
    table_name_map: Dict[str, str] = {}
    for t in tables.keys():
        table_name_map.setdefault(_norm(t), t)

    for tc in relevant_columns:
        if "." not in tc:
            continue
        raw_table_name, raw_column_name = tc.split(".", 1)
        table_name = table_name_map.get(_norm(raw_table_name))
        if not table_name:
            continue

        cols = tables[table_name].get("columns") or {}
        if raw_column_name in cols:
            linked.setdefault(table_name, [])
            if raw_column_name not in linked[table_name]:
                linked[table_name].append(raw_column_name)
            continue

        # Case-insensitive fallback: match by schema key or by stored column_name.
        target_col_norm = _norm(raw_column_name)
        matched: Optional[str] = None
        for actual_col_name, col_info in cols.items():
            if _norm(actual_col_name) == target_col_norm:
                matched = actual_col_name
                break
            if (
                isinstance(col_info, dict)
                and _norm(col_info.get("column_name", "")) == target_col_norm
            ):
                matched = actual_col_name
                break
        if matched:
            linked.setdefault(table_name, [])
            if matched not in linked[table_name]:
                linked[table_name].append(matched)

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


def _is_filter_fragment(frag: str) -> bool:
    """Return True if the fragment is a filter predicate (contains operator + value)."""
    # A fragment is a filter if it contains a comparison operator followed by a value
    # e.g. '"col" = \'val\'', '"col" > 5', '"col" LIKE \'%x%\''
    filter_pattern = re.compile(
        r"""(?:=|!=|<>|>=|<=|>|<|\bLIKE\b|\bIN\b|\bIS\b)\s*(?:'[^']*'|\d+|\(|\bNULL\b)""",
        re.IGNORECASE,
    )
    return bool(filter_pattern.search(frag))


def _format_structural_join_block(i: int, uc: dict) -> str:
    """Format a structural_join user_choice entry."""
    tables = uc.get("tables") or []
    no_join = uc.get("no_join", False)
    chosen_columns = uc.get("chosen_columns") or []
    chosen_fragment = uc.get("chosen_fragment") or ""
    t0 = tables[0] if len(tables) > 0 else "?"
    t1 = tables[1] if len(tables) > 1 else "?"

    if no_join:
        return (
            f"{i}. `{t0}` and `{t1}` do not need to be joined."
        )
    else:
        cond = f"`{chosen_fragment}`" if chosen_fragment else (
            f"`{' = '.join(chosen_columns)}`" if chosen_columns else "(see schema)"
        )
        return f"{i}. `{t0}` JOIN `{t1}` on {cond} ✓"


def _format_semantic_join_block(i: int, uc: dict) -> str:
    """Format a semantic_join user_choice entry."""
    semantic_tables = uc.get("semantic_tables") or []
    routing_tables = uc.get("routing_tables") or []
    chosen_join_conditions = uc.get("chosen_join_conditions") or []
    no_join = uc.get("no_join", False)

    t0 = semantic_tables[0] if len(semantic_tables) > 0 else "?"
    t1 = semantic_tables[-1] if len(semantic_tables) > 1 else "?"

    if no_join:
        return f"{i}. `{t0}` and `{t1}` — **do not join**."

    lines: List[str] = []
    if routing_tables:
        path_str = " → ".join([t0] + routing_tables + [t1])
        lines.append(
            f"{i}. To connect `{t0}` and `{t1}`, route through `{'`, `'.join(routing_tables)}` "
            f"(implicit bridge table{'s' if len(routing_tables) > 1 else ''} not mentioned in the question — must include)."
        )
        lines.append(f"   Full path: {path_str}")
    else:
        lines.append(f"{i}. `{t0}` and `{t1}` are joined directly.")

    if chosen_join_conditions:
        conds_str = ", ".join(f"`{c}`" for c in chosen_join_conditions)
        lines.append(f"   Join conditions: {conds_str}")

    return "\n".join(lines)


def format_offline_memory_mapping_hint(record: dict) -> str:
    """
    Build mapping_hint from user_choices (exact JSON fields) + guidance +
    guidance_mapping_nodes. Does not include historical_pairs.
    """
    parts: List[str] = []
    user_choices = record.get("user_choices") or []
    guidance = record.get("guidance") or []
    guidance_nodes = record.get("guidance_mapping_nodes") or []

    # Separate user_choices by type
    column_mapping_choices = [
        uc for uc in user_choices
        if isinstance(uc, dict) and uc.get("mention_type") not in ("structural_join", "semantic_join")
    ]
    structural_join_choices = [
        uc for uc in user_choices
        if isinstance(uc, dict) and uc.get("mention_type") == "structural_join"
    ]
    semantic_join_choices = [
        uc for uc in user_choices
        if isinstance(uc, dict) and uc.get("mention_type") == "semantic_join"
    ]

    if column_mapping_choices:
        # Build header: only mention implicit_conditions / mapping_nodes if they actually appear
        has_implicit = any(
            any(mn.get("implicit_conditions") for mn in (uc.get("mapping_nodes") or []) if isinstance(mn, dict))
            for uc in column_mapping_choices if isinstance(uc, dict)
        )
        has_mapping_nodes = any(
            bool(uc.get("mapping_nodes"))
            for uc in column_mapping_choices if isinstance(uc, dict)
        )

        header_lines = [
            "## User selection (column mapping hints — highest priority)",
            "The user identified which database columns correspond to specific mentions in the question. "
            "These are **partial mapping hints only** — they do NOT tell you what to SELECT or how to filter. "
            "Use your own reasoning to construct the full SQL.",
            "",
            "**Fields:**",
            "- `columns` (**gold columns — mandatory**): confirmed columns for this mention. "
            "MUST appear in your SQL (as JOIN key, filter, or aggregate — depends on context).",
            "- `sql_fragments` (structural template — follow the pattern, do NOT copy literal values).",
        ]
        if has_mapping_nodes:
            header_lines.append(
                "- `mapping_nodes` (historical column co-occurrence — use only to understand join paths; "
                "ignore all filter values and operators)."
            )
        if has_implicit:
            header_lines.append(
                "- `implicit_conditions` (**mandatory** — user-confirmed conditions not stated in the question "
                "but required by business logic; include them in your SQL as-is)."
            )
        parts.append("\n".join(header_lines))
        for i, uc in enumerate(column_mapping_choices, 1):
            if not isinstance(uc, dict):
                continue
            mention = uc.get("mention_text", "")
            focus = uc.get("focus") or ""
            cols = uc.get("columns") or []
            op_sig = uc.get("operation_signature") or {}
            op_keywords = op_sig.get("sql_keywords") or []

            block_lines = [
                f'{i}. **mention_text:** "{mention}"',
            ]
            if focus:
                block_lines.append(f'   - **focus (core concept):** "{focus}"')
            block_lines.append(f"   - **columns (gold — MUST use; role depends on context):** {cols}")
            if op_keywords:
                block_lines.append(
                    f"   - **operation_signature (required SQL operations):** {op_keywords}"
                )

            # Direct sql_fragments: filter out pure filter predicates (enum values / specific literals)
            # to avoid misleading the model with historical values
            direct_frags_raw: List[str] = list(uc.get("sql_fragments") or [])
            direct_frags_col = [f for f in direct_frags_raw if not _is_filter_fragment(f)]
            direct_frags_filter = [f for f in direct_frags_raw if _is_filter_fragment(f)]

            implicit_all: List[str] = []
            node_column_frags: List[str] = []  # only column-ref frags from mapping_nodes (no filters)
            node_detail_lines: List[str] = []
            for mn in uc.get("mapping_nodes") or []:
                if not isinstance(mn, dict):
                    continue
                mid = mn.get("mapping_node_id", "")
                mn_implicit = mn.get("implicit_conditions") or []
                for ic in mn_implicit:
                    implicit_all.append(ic)
                # From mapping_nodes: only keep non-filter fragments (column refs / function calls)
                mn_frags_all = mn.get("sql_fragments") or []
                mn_frags_col = [f for f in mn_frags_all if not _is_filter_fragment(f)]
                node_column_frags.extend(mn_frags_col)
                col_frags_str = mn_frags_col if mn_frags_col else "(none — all were filter predicates, ignored)"
                node_detail_lines.append(
                    f'     - **mapping_node_id** `{mid}` (join/co-occurrence pattern only): '
                    f'column refs={col_frags_str}'
                    + (f', implicit_conditions={mn_implicit} ← **MUST apply these conditions**' if mn_implicit else "")
                )

            direct_frags_col = _dedupe_preserve_order(direct_frags_col)
            direct_frags_filter = _dedupe_preserve_order(direct_frags_filter)
            node_column_frags = _dedupe_preserve_order(node_column_frags)
            implicit_all = _dedupe_preserve_order(implicit_all)

            if direct_frags_col:
                block_lines.append(
                    f"   - **sql_fragments (operation structure — do NOT copy values):** {direct_frags_col}"
                )
            # Show filter-type fragments separately with explicit warning
            if direct_frags_filter:
                block_lines.append(
                    f"   - **sql_fragments (filter examples from history — operation structure ONLY; "
                    f"re-derive all filter values from the current question):** {direct_frags_filter}"
                )
            if implicit_all:
                block_lines.append(
                    f"   - **implicit_conditions (user-confirmed mandatory constraints — MUST include in SQL):** "
                    f"{implicit_all}"
                )
            if node_detail_lines:
                block_lines.append(
                    "   - **mapping_nodes (column co-occurrence from history — ignore all filter values/operators):**"
                )
                block_lines.extend(node_detail_lines)
            parts.append("\n".join(block_lines))

    if structural_join_choices or semantic_join_choices:
        parts.append(
            "## Join path (user-confirmed — must follow strictly)\n"
            "The join path below was verified by the user. "
            "Follow it exactly — do not substitute, skip, or add tables based on your own inference."
        )

        join_idx = 1
        for uc in structural_join_choices:
            parts.append(_format_structural_join_block(join_idx, uc))
            join_idx += 1
        for uc in semantic_join_choices:
            parts.append(_format_semantic_join_block(join_idx, uc))
            join_idx += 1

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
            block: List[str] = [
                f"{gidx}. **target mention:** `{mt}`"
                if mt
                else f"{gidx}. **target mention:** *(empty)*"
            ]
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
                    [
                        str(gn.get("mapping_node_id"))
                        for gn in guidance_nodes
                        if isinstance(gn, dict) and gn.get("mapping_node_id")
                    ]
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
        "The following are question–SQL pairs from the same database, retrieved because they share "
        "similar schema access patterns with the current question. "
        "**These are partial hints, not complete solutions.** Use them to understand "
        "**which columns and tables tend to appear together**, **how joins are structured**, and "
        "**what SQL operations are typically applied** for similar semantic concepts. "
        "Do NOT copy the SQL directly — re-derive all conditions, filters, and values from the current question.",
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


def format_wo_interaction_guidance_hint(record: dict, include_reasoning: bool = False) -> str:
    """
    Format guidance from context graph memory without user interaction signals.

    This guidance is derived from historical Q&A patterns and represents
    preference signals, not absolute correctness.
    """
    guidance_content = record.get(
        "guidance_content_with_reasoning" if include_reasoning else "guidance_content_no_reasoning"
    )
    if not guidance_content:
        return ""

    header = """## Guidance Methodology

This guidance is derived from historical Q&A patterns from similar questions.
Each guidance item indicates either:
  (a) strong_preference - statistical dominance from historical evidence
  (b) error_warning - historical mistake pattern to avoid
  (c) ambiguity_note - genuine ambiguity with multiple viable options

**Important**: This guidance represents historical user preferences and patterns,
NOT absolute correct answers. You must analyze the current question's specific
context, constraints, and requirements before applying any guidance. Historical
patterns may not apply to the current situation."""

    return f"{header}\n\n{guidance_content}"


def augment_schema_with_memory_columns(
    base_schema: Dict[str, Any],
    linked: Dict[str, List[str]],
    full_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Augment base_schema (e.g. database_schema_after_schema_linking) by adding any
    memory-required columns that are missing from it.

    - Tables in `linked` that already exist in base_schema: add missing columns from full_schema.
    - Tables in `linked` that are NOT in base_schema: add the entire table from full_schema.
    - Tables already in base_schema but not in `linked`: kept as-is (no reduction).

    Returns a new schema dict (shallow copy of base_schema with additions applied).
    """
    augmented = copy.deepcopy(base_schema)
    full_tables = full_schema.get("tables") or {}

    for table_name, memory_cols in linked.items():
        full_table = full_tables.get(table_name)
        if not full_table:
            continue

        if table_name not in augmented["tables"]:
            # Table not in base schema at all — add it from full_schema
            augmented["tables"][table_name] = copy.deepcopy(full_table)
        else:
            # Table exists — add missing columns
            existing_cols = augmented["tables"][table_name].get("columns") or {}
            full_cols = full_table.get("columns") or {}
            for col_name in memory_cols:
                if col_name not in existing_cols and col_name in full_cols:
                    existing_cols[col_name] = copy.deepcopy(full_cols[col_name])
            augmented["tables"][table_name]["columns"] = existing_cols

    return augmented


def _mapping_hint_from_offline_record(record: dict) -> Optional[str]:
    """Use the JSONL record's pre-written mapping_hint string; no runtime formatting."""
    raw = record.get("mapping_hint")
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    return s if s else None


def ensure_mapping_data(
    data_item: DataItem,
    path: Optional[str] = None,
    format_type: str = "with_user_interaction",
    include_reasoning: bool = False,
) -> None:
    """
    Populate mapping_* or guidance_hint fields from offline context-graph memory jsonl for this question_id
    and database_id.

    format_type:
      - "with_user_interaction": Use user_choices, guidance, guidance_mapping_nodes, historical_pairs
      - "wo_user_interaction": Use guidance_content_{with,no}_reasoning only
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

    if format_type == "wo_user_interaction":
        # New format: only guidance_content, no user_choices
        guidance_hint = format_wo_interaction_guidance_hint(record, include_reasoning)
        if guidance_hint:
            data_item.guidance_hint = guidance_hint
            logger.debug(
                "[Mapping] Populated guidance_hint for question_id=%s (wo_user_interaction format)",
                data_item.question_id,
            )
        # mapping_hint and related fields remain None
        return

    # Existing format: with_user_interaction (default)
    relevant_columns = _relevant_columns_from_record(record)
    # Use the full (value-retrieval) schema as the source of truth for column metadata
    full_schema = data_item.database_schema_after_value_retrieval or data_item.database_schema

    if relevant_columns:
        linked = relevant_columns_to_linked_schema(relevant_columns, full_schema)
        if linked:
            data_item.mapping_linked_tables_and_columns = linked

            # Augment schema_linking result with any memory-required columns/tables
            # that are missing from it — instead of replacing it with a restricted subset.
            base_schema = (
                data_item.database_schema_after_schema_linking
                or full_schema
            )
            data_item.database_schema_after_mapping = augment_schema_with_memory_columns(
                base_schema, linked, full_schema
            )
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
    data_item.mapping_historical_qa = historical_pairs_to_mapping_qa(
        record.get("historical_pairs") or []
    )

    if not data_item.mapping_hint and not data_item.mapping_historical_qa:
        data_item.mapping_linked_tables_and_columns = None
        data_item.database_schema_after_mapping = None

