import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.dataset import DataItem
from app.logger import logger

from .base import MemoryAugmentor


class AMemRetrievalMemoryAugmentor(MemoryAugmentor):
    name = "amem_retrieval"

    _records_cache: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}

    def _get_data_path(self) -> Optional[str]:
        data_path = self.strategy_config.get("data_path")
        if data_path:
            return data_path
        return self.strategy_config.get("index_path")

    def supports(self, data_item: DataItem) -> bool:
        data_path = self._get_data_path()
        return bool(data_path and Path(data_path).exists())

    def _load_index(self, data_path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
        resolved_path = str(Path(data_path).resolve())
        cached = self._records_cache.get(resolved_path)
        if cached is not None:
            return cached

        path_obj = Path(resolved_path)
        suffix = path_obj.suffix.lower()
        if suffix not in {".jsonl", ".json"}:
            raise ValueError(f"Unsupported A-mem retrieval file format: {resolved_path}")

        if suffix == ".jsonl":
            with path_obj.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with path_obj.open("r", encoding="utf-8") as f:
                rows = json.load(f)

        index: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for row in rows:
            db_id = str(row.get("db_id", ""))
            question_id = row.get("question_id")
            if question_id is None:
                continue
            key = (db_id, int(question_id))
            index[key] = row

        self._records_cache[resolved_path] = index
        return index

    @staticmethod
    def _parse_note_content(content: str) -> Dict[str, str]:
        """
        Parse multi-line A-mem note content into structured fields.
        Expected labels include Task/Database/Question/Evidence/SQL/Context/Keywords.
        """
        parsed: Dict[str, str] = {}
        current_key: Optional[str] = None

        field_map = {
            "task": "task",
            "database": "database",
            "question": "question",
            "evidence": "evidence",
            "sql": "sql",
            "context": "context",
            "keywords": "keywords",
        }

        for raw_line in (content or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            matched_new_field = False
            for prefix, target_key in field_map.items():
                tag = f"{prefix}:"
                if line.lower().startswith(tag):
                    value = line[len(tag):].strip()
                    parsed[target_key] = value
                    current_key = target_key
                    matched_new_field = True
                    break

            if matched_new_field:
                continue

            # Support wrapped lines for the most recently seen field.
            if current_key:
                existing = parsed.get(current_key, "")
                parsed[current_key] = f"{existing}\n{line}".strip()

        return parsed

    def augment(self, data_item: DataItem) -> None:
        data_path = self._get_data_path()
        if not data_path:
            return

        index = self._load_index(data_path)
        key = (str(getattr(data_item, "database_id", "")), int(getattr(data_item, "question_id", -1)))
        row = index.get(key)

        metadata = dict(getattr(data_item, "memory_metadata", None) or {})
        if row is None:
            metadata[self.name] = {
                "enabled": True,
                "hit": False,
                "data_path": str(Path(data_path)),
                "key": {"db_id": key[0], "question_id": key[1]},
            }
            data_item.memory_metadata = metadata
            logger.info(
                "[amem_retrieval] No memory row found for question_id={} db={}",
                key[1],
                key[0],
            )
            return

        retrieved = row.get("retrieved") or []
        configured_top_k = int(self.strategy_config.get("top_k", row.get("retrieval_k", len(retrieved))))
        if configured_top_k <= 0:
            selected = []
        else:
            selected = retrieved[:configured_top_k]

        memory_items = list(getattr(data_item, "memory_items", None) or [])
        qa_lines = ["## Reference: Retrieved Historical Question-SQL Pairs"]
        strategy_items: List[Dict[str, Any]] = []

        for fallback_rank, retrieved_item in enumerate(selected, start=1):
            parsed = self._parse_note_content(str(retrieved_item.get("content", "")))
            rank = int(retrieved_item.get("rank", fallback_rank))
            question = parsed.get("question", "")
            sql = parsed.get("sql", "")
            evidence = parsed.get("evidence", "")
            raw_content = str(retrieved_item.get("content", "")).strip()

            item: Dict[str, Any] = {
                "strategy": self.name,
                "type": "qa_pair",
                "rank": rank,
                "note_id": retrieved_item.get("note_id"),
                "memory_index": retrieved_item.get("memory_index"),
                "database_id": row.get("db_id"),
                "question": question,
                "sql": sql,
                "evidence": evidence,
                "raw_content": raw_content,
            }
            strategy_items.append(item)
            memory_items.append(item)

            if question and sql:
                qa_lines.append(f"- Example {rank}: Question: {question}")
                qa_lines.append(f"  SQL: {sql}")
            else:
                qa_lines.append(f"- Example {rank}:")
                qa_lines.append(f"  {raw_content}")

        data_item.memory_items = memory_items

        summary_blocks = []
        existing_summary = getattr(data_item, "memory_summary", None)
        if existing_summary:
            summary_blocks.append(existing_summary)
        if strategy_items:
            summary_blocks.append("\n".join(qa_lines).strip())
        data_item.memory_summary = "\n\n".join([x for x in summary_blocks if x]).strip()

        metadata[self.name] = {
            "enabled": True,
            "hit": True,
            "data_path": str(Path(data_path)),
            "key": {"db_id": key[0], "question_id": key[1]},
            "retrieval_k": int(row.get("retrieval_k", len(retrieved))),
            "retrieved_count": int(row.get("retrieved_count", len(retrieved))),
            "n_candidates": len(strategy_items),
            "top_k": configured_top_k,
        }
        data_item.memory_metadata = metadata

        logger.info(
            "[amem_retrieval] question_id={} db={} loaded {} memory items",
            key[1],
            key[0],
            len(strategy_items),
        )
