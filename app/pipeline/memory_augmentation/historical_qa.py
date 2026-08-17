import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.dataset import DataItem
from app.logger import logger
from app.vector_db.vector_db import get_embedding_function

from .base import MemoryAugmentor


class HistoricalQAMemoryAugmentor(MemoryAugmentor):
    name = "historical_qa"

    _records_cache: Dict[str, List[Dict[str, Any]]] = {}
    _embeddings_cache: Dict[str, np.ndarray] = {}
    _embedding_function = None

    def _get_data_path(self) -> Optional[str]:
        data_path = self.strategy_config.get("data_path")
        if data_path:
            return data_path
        return self.strategy_config.get("index_path")

    def supports(self, data_item: DataItem) -> bool:
        data_path = self._get_data_path()
        return bool(data_path and Path(data_path).exists())

    def _load_records(self, data_path: str) -> List[Dict[str, Any]]:
        path = str(Path(data_path).resolve())
        cached = self._records_cache.get(path)
        if cached is not None:
            return cached
        suffix = Path(path).suffix.lower()
        if suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            raise ValueError(f"Unsupported historical QA file format: {path}")
        self._records_cache[path] = records
        return records

    def _get_embedding_function(self):
        if self._embedding_function is not None:
            return self._embedding_function
        model_name_or_path = self.strategy_config.get("embedding_model_name_or_path")
        if not model_name_or_path:
            model_name_or_path = self.strategy_config.get("embedding_model")
        if not model_name_or_path:
            raise ValueError("historical_qa strategy requires embedding_model_name_or_path or embedding_model")
        self._embedding_function = get_embedding_function(
            model_name_or_path=model_name_or_path,
            use_qwen3_embedding=self.strategy_config.get("use_qwen3_embedding", False),
            local_files_only=self.strategy_config.get("local_files_only", False),
            normalize_embeddings=self.strategy_config.get("normalize_embeddings", False),
            device=self.strategy_config.get("device", "cpu"),
            base_url=self.strategy_config.get("base_url"),
            api_key=self.strategy_config.get("api_key"),
        )
        return self._embedding_function

    def _build_embeddings(self, records: List[Dict[str, Any]], data_path: str) -> np.ndarray:
        resolved = str(Path(data_path).resolve())
        cached = self._embeddings_cache.get(resolved)
        if cached is not None:
            return cached

        if records and isinstance(records[0].get("question_embedding"), list):
            embeddings = np.asarray([r.get("question_embedding", []) for r in records], dtype=float)
            self._embeddings_cache[resolved] = embeddings
            return embeddings

        embedding_path = self.strategy_config.get("embedding_path")
        if embedding_path and Path(embedding_path).exists():
            embeddings = np.load(embedding_path)
            self._embeddings_cache[resolved] = embeddings
            return embeddings

        embed_fn = self._get_embedding_function()
        questions = [str(r.get("question", "")) for r in records]
        embeddings = np.asarray(embed_fn(questions), dtype=float)
        self._embeddings_cache[resolved] = embeddings
        return embeddings

    @staticmethod
    def _cosine_similarity(query_embedding: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_embedding)
        candidate_norm = np.linalg.norm(candidates, axis=1)
        denom = np.clip(candidate_norm * query_norm, a_min=1e-12, a_max=None)
        return (candidates @ query_embedding) / denom

    def _retrieve_top_k(
        self,
        data_item: DataItem,
        records: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> List[Tuple[int, float]]:
        use_same_database = self.strategy_config.get("same_database_only", True)
        candidate_indices = [
            idx
            for idx, row in enumerate(records)
            if (not use_same_database) or row.get("database_id") == data_item.database_id
        ]
        if not candidate_indices:
            return []

        embed_fn = self._get_embedding_function()
        query_embedding = np.asarray(embed_fn([data_item.question])[0], dtype=float)
        candidate_embeddings = embeddings[candidate_indices]
        similarities = self._cosine_similarity(query_embedding, candidate_embeddings)

        top_k = int(self.strategy_config.get("top_k", 5))
        if top_k <= 0:
            return []
        top_rel_indices = np.argsort(similarities)[::-1][:top_k]
        return [(candidate_indices[i], float(similarities[i])) for i in top_rel_indices]

    def augment(self, data_item: DataItem) -> None:
        data_path = self._get_data_path()
        if not data_path:
            return
        records = self._load_records(data_path)
        if not records:
            return
        embeddings = self._build_embeddings(records, data_path)
        if len(embeddings) != len(records):
            logger.warning(
                "[historical_qa] embedding count does not match records: %s vs %s",
                len(embeddings),
                len(records),
            )
            return

        top_matches = self._retrieve_top_k(data_item, records, embeddings)
        if not top_matches:
            logger.info(
                "[historical_qa] No QA retrieved for question_id=%s, db=%s, question=%r",
                getattr(data_item, "question_id", None),
                getattr(data_item, "database_id", None),
                getattr(data_item, "question", None),
            )
            return

        # Be robust to older pickled DataItem objects that may not yet have
        # the memory_* fields populated on load.
        memory_items = list(getattr(data_item, "memory_items", None) or [])
        qa_lines = ["## Reference: Retrieved Historical Question-SQL Pairs"]
        strategy_items: List[Dict[str, Any]] = []
        for rank, (record_idx, score) in enumerate(top_matches, start=1):
            row = records[record_idx]
            item = {
                "strategy": self.name,
                "type": "qa_pair",
                "rank": rank,
                "similarity": score,
                "database_id": row.get("database_id"),
                "question": row.get("question", ""),
                "sql": row.get("sql", ""),
                "qa_id": row.get("qa_id"),
            }
            strategy_items.append(item)
            memory_items.append(item)
            qa_lines.append(f"- Example {rank}: Question: {item['question']}")
            qa_lines.append(f"  SQL: {item['sql']}")

        # Debug logging: print retrieved QA pairs for inspection (first few only)
        try:
            logger.info(
                "[historical_qa] question_id=%s db=%s retrieved %d QA pairs, top_similarity=%.4f",
                getattr(data_item, "question_id", None),
                getattr(data_item, "database_id", None),
                len(strategy_items),
                strategy_items[0]["similarity"] if strategy_items else float("nan"),
            )
            for item in strategy_items[:5]:
                msg = (
                    "[historical_qa]   "
                    f"rank={item['rank']} "
                    f"sim={item['similarity']:.4f} "
                    f"qa_id={item.get('qa_id')} "
                    f"question={item.get('question')!r} "
                    f"sql={item.get('sql')!r}"
                )
                logger.info(msg)
        except Exception:
            # Logging should never break the main pipeline; ignore logging errors.
            pass

        data_item.memory_items = memory_items
        summary_blocks = []
        existing_summary = getattr(data_item, "memory_summary", None)
        if existing_summary:
            summary_blocks.append(existing_summary)
        summary_blocks.append("\n".join(qa_lines).strip())
        data_item.memory_summary = "\n\n".join([x for x in summary_blocks if x]).strip()

        metadata = dict(getattr(data_item, "memory_metadata", None) or {})
        metadata[self.name] = {
            "enabled": True,
            "n_candidates": len(strategy_items),
            "top_similarity": strategy_items[0]["similarity"],
            "top_k": int(self.strategy_config.get("top_k", 5)),
            "data_path": str(Path(data_path)),
        }
        data_item.memory_metadata = metadata
