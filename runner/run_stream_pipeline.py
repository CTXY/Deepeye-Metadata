#!/usr/bin/env python3
"""
Stream-like pipeline runner.

Goal:
- Process data item by item (or by small batches), running each enabled pipeline stage sequentially.
- After each item finishes the last stage (SQL selection), immediately evaluate current Execution Accuracy (EA)
  on processed items so far.

Design constraints:
- Minimize changes to existing project code: this is a new runner that reuses existing stage implementations.
- Reuse the same internal stage logic (generators/checkers/selection) by calling the existing per-item methods.

Usage examples:
  python runner/run_stream_pipeline.py --start-from memory_augmentation --question-id-ranges  44-88 435-530
  python runner/run_stream_pipeline.py --start-from sql_generation --question-id-ranges 44-88 435-530 142-194 624-716
  python runner/run_stream_pipeline.py --start-from value_retrieval --batch-size 4 --save-every 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

sys.path.append(".")

from app.config import config
from app.dataset import DataItem, load_dataset, save_dataset
from app.db_utils.execution import execute_sql
from app.llm import LLM
from app.logger import logger


def _parse_question_id_ranges(ranges: List[str]) -> Set[int]:
    ids: Set[int] = set()
    for s in ranges:
        s = s.strip()
        if not s:
            continue
        if "-" in s:
            low_s, high_s = s.split("-", 1)
            low, high = int(low_s.strip()), int(high_s.strip())
            ids.update(range(low, high + 1))
        else:
            ids.add(int(s))
    return ids


def _iter_batches(items: List[DataItem], batch_size: int) -> Iterable[List[DataItem]]:
    if batch_size <= 1:
        for x in items:
            yield [x]
        return
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _eval_ex(pred_sql: str, gold_sql: str, db_path: str) -> Optional[int]:
    """
    Same evaluation semantics as script/calculate_ex_accuracy.py:
    - If gold SQL fails to execute: return None
    - If pred fails: return 0
    - Otherwise compare results ignoring order (using row-wise frozenset to match existing behavior)
    """
    pred_result = execute_sql(db_path, pred_sql)
    gold_result = execute_sql(db_path, gold_sql)

    if gold_result.result_rows is None:
        return None
    if pred_result.result_rows is None:
        return 0

    pred_result_set = set(map(frozenset, pred_result.result_rows))
    gold_result_set = set(map(frozenset, gold_result.result_rows))
    return 1 if pred_result_set == gold_result_set else 0


class _StreamStages:
    """
    Lightweight stage container that reuses existing per-item implementations
    without forcing the original stage runners to load/save whole datasets.
    """

    def __init__(self):
        # Value retrieval stage: reuse ValueRetrievalRunner internals (vector DB init etc.)
        from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner

        self.value_retrieval = ValueRetrievalRunner.__new__(ValueRetrievalRunner)
        self.value_retrieval._llm = LLM(config.value_retrieval_config.llm)
        self.value_retrieval._dataset = None
        self.value_retrieval._vector_db_client_dict = {}
        self.value_retrieval._vector_db_collection_dict = {}
        self.value_retrieval._thread_pool_executor = None

        # Schema linking stage: reuse SchemaLinkingRunner internal method
        from app.pipeline.schema_linking.schema_linking import SchemaLinkingRunner

        self.schema_linking = SchemaLinkingRunner.__new__(SchemaLinkingRunner)
        self.schema_linking._llm = LLM(config.schema_linking_config.llm)
        self.schema_linking._dataset = None
        self.schema_linking._thread_pool_executor = None
        from app.pipeline.schema_linking.linkers import DirectLinker, ReversedLinker, ValueLinker

        self.schema_linking._direct_linker = DirectLinker()
        self.schema_linking._reversed_linker = ReversedLinker()
        self.schema_linking._value_linker = ValueLinker()
        self.schema_linking._caf_system = None
        # mirror init logic for CAF
        if config.schema_linking_config.use_caf_metadata:
            try:
                import caf  # type: ignore

                caf_config_path = Path("config/caf_config.yaml")
                if caf_config_path.exists():
                    self.schema_linking._caf_system = caf.initialize(config_path=str(caf_config_path))
                    logger.info("CAF system initialized for stream schema linking")
                else:
                    logger.warning("CAF config not found; stream schema linking will not use CAF metadata")
            except Exception as e:
                logger.warning(f"CAF unavailable for stream schema linking: {e}")

        # Memory augmentation stage: reuse strategy registry directly
        from app.pipeline.memory_augmentation.registry import get_augmentors_from_config

        self.memory_augmentors = get_augmentors_from_config()

        # SQL generation stage: reuse SQLGenerationRunner internal method
        from app.pipeline.sql_generation.sql_generation import SQLGenerationRunner
        from app.pipeline.sql_generation.generators import DCGenerator, SkeletonGenerator, ICLGenerator

        self.sql_generation = SQLGenerationRunner.__new__(SQLGenerationRunner)
        self.sql_generation._llm = LLM(config.sql_generation_config.llm)
        self.sql_generation._dataset = None
        self.sql_generation._thread_pool_executor = None
        self.sql_generation._dc_generator = DCGenerator()
        self.sql_generation._skeleton_generator = SkeletonGenerator()
        self.sql_generation._icl_generator = ICLGenerator()

        # SQL revision stage: reuse SQLRevisionRunner internal method
        from app.pipeline.sql_revision.sql_revision import SQLRevisionRunner
        from app.pipeline.sql_revision.checkers import (
            JoinChecker,
            MaxMinChecker,
            OrderByLimitChecker,
            OrderByNullChecker,
            ResultChecker,
            SelectChecker,
            SyntaxChecker,
            TimeChecker,
        )

        self.sql_revision = SQLRevisionRunner.__new__(SQLRevisionRunner)
        self.sql_revision._llm = LLM(config.sql_revision_config.llm)
        self.sql_revision._dataset = None
        self.sql_revision._thread_pool_executor = None
        self.sql_revision._checkers = [
            SyntaxChecker(),
            JoinChecker(),
            OrderByLimitChecker(),
            TimeChecker(),
            SelectChecker(),
            MaxMinChecker(),
            OrderByNullChecker(),
            ResultChecker(),
        ]

        # SQL selection stage: reuse BRSelectionRunner internal method
        from app.pipeline.sql_selection.br_selection import BRSelectionRunner

        self.sql_selection = BRSelectionRunner.__new__(BRSelectionRunner)
        self.sql_selection._llm = LLM(config.sql_selection_config.llm)
        self.sql_selection._dataset = None
        self.sql_selection._thread_pool_executor = None

    def close(self):
        # Cleanup CAF if used
        caf_system = getattr(self.schema_linking, "_caf_system", None)
        if caf_system is not None:
            try:
                caf_system.cleanup()
            except Exception:
                pass

    def ensure_value_retrieval_collections(self, database_ids: Set[str]) -> None:
        """
        Lazy init chroma collections only for databases that appear in the stream subset.
        """
        from chromadb import PersistentClient
        from app.vector_db import get_embedding_function

        for db_id in sorted(database_ids):
            if db_id in self.value_retrieval._vector_db_collection_dict:
                continue
            vector_db_path = Path(config.vector_database_config.store_root_path) / db_id
            client = PersistentClient(path=vector_db_path)
            collection = client.get_collection(
                name=db_id,
                embedding_function=get_embedding_function(
                    model_name_or_path=config.vector_database_config.embedding_model_name_or_path,
                    use_qwen3_embedding=config.vector_database_config.use_qwen3_embedding,
                    local_files_only=config.vector_database_config.local_files_only,
                    normalize_embeddings=config.vector_database_config.normalize_embeddings,
                    device=config.vector_database_config.device,
                    base_url=config.vector_database_config.base_url,
                    api_key=config.vector_database_config.api_key,
                ),
            )
            self.value_retrieval._vector_db_client_dict[db_id] = client
            self.value_retrieval._vector_db_collection_dict[db_id] = collection

        # Ensure thread pool exists (used to parallelize per-column retrieval within one item)
        if self.value_retrieval._thread_pool_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self.value_retrieval._thread_pool_executor = ThreadPoolExecutor(
                max_workers=config.value_retrieval_config.n_parallel
            )

    def run_value_retrieval(self, data_item: DataItem) -> None:
        # Ensure collection for this db exists
        self.ensure_value_retrieval_collections({data_item.database_id})

        start = time.time()
        self.value_retrieval._retrieve_values(data_item)
        data_item.value_retrieval_time = time.time() - start
        data_item.total_time = data_item.value_retrieval_time
        data_item.total_llm_cost = data_item.value_retrieval_llm_cost

    def run_schema_linking(self, data_item: DataItem) -> None:
        self.schema_linking._link_tables_and_columns(data_item)

    def run_memory_augmentation(self, data_item: DataItem) -> None:
        # Same logic as MemoryAugmentationRunner._augment_one but without needing to load dataset.
        start_time = time.time()

        memory_metadata = getattr(data_item, "memory_metadata", None) or {}
        setattr(data_item, "memory_metadata", memory_metadata)
        memory_metadata.setdefault(
            "enabled_strategies",
            [augmentor.name for augmentor in self.memory_augmentors],
        )

        for augmentor in self.memory_augmentors:
            if not augmentor.supports(data_item):
                continue
            augmentor.augment(data_item)

        memory_augmentation_time = time.time() - start_time
        setattr(data_item, "memory_augmentation_time", memory_augmentation_time)

        _zero_llm_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        memory_augmentation_llm_cost = getattr(data_item, "memory_augmentation_llm_cost", None) or dict(
            _zero_llm_cost
        )
        setattr(data_item, "memory_augmentation_llm_cost", memory_augmentation_llm_cost)

        total_time = getattr(data_item, "total_time", None) or 0.0
        setattr(data_item, "total_time", total_time + memory_augmentation_time)

        current_total_llm_cost = getattr(data_item, "total_llm_cost", None) or dict(_zero_llm_cost)
        updated_total_llm_cost = {
            "prompt_tokens": current_total_llm_cost.get("prompt_tokens", 0)
            + memory_augmentation_llm_cost.get("prompt_tokens", 0),
            "completion_tokens": current_total_llm_cost.get("completion_tokens", 0)
            + memory_augmentation_llm_cost.get("completion_tokens", 0),
            "total_tokens": current_total_llm_cost.get("total_tokens", 0)
            + memory_augmentation_llm_cost.get("total_tokens", 0),
        }
        setattr(data_item, "total_llm_cost", updated_total_llm_cost)

    def run_sql_generation(self, data_item: DataItem) -> None:
        self.sql_generation._generate_sql(data_item)

    def run_sql_revision(self, data_item: DataItem) -> None:
        self.sql_revision._revise_sql(data_item)

    def run_sql_selection(self, data_item: DataItem) -> None:
        self.sql_selection._select_best_sql(data_item)


def _resolve_start_dataset_path(start_from: str) -> str:
    mapping = {
        "dataset": config.dataset_config.save_path,
        "value_retrieval": config.value_retrieval_config.save_path,
        "schema_linking": config.schema_linking_config.save_path,
        "memory_augmentation": config.memory_augmentation_config.save_path,
        "sql_generation": config.sql_generation_config.save_path,
        "sql_revision": config.sql_revision_config.save_path,
        "sql_selection": config.sql_selection_config.save_path,
    }
    return mapping[start_from]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEye-SQL pipeline in stream mode (per item/batch).")
    parser.add_argument(
        "--start-from",
        type=str,
        default="dataset",
        choices=[
            "dataset",
            "value_retrieval",
            "schema_linking",
            "memory_augmentation",
            "sql_generation",
            "sql_revision",
            "sql_selection",
        ],
        help="Which stage output pickle to load as the starting point.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N items after filtering.")
    parser.add_argument("--batch-size", type=int, default=1, help="Process items in small batches (default: 1).")
    parser.add_argument(
        "--question-id-ranges",
        type=str,
        nargs="+",
        default=None,
        metavar="RANGE",
        help="Only process these question_id ranges (inclusive), e.g. 44-88 435-530",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save intermediate dataset/result files every N processed items (default: 20).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(Path(config.sql_selection_config.save_path).with_suffix(".stream.json")),
        help="Where to write streaming predicted SQL mapping {question_id: sql}.",
    )
    args = parser.parse_args()

    start_path = _resolve_start_dataset_path(args.start_from)
    logger.info(f"[stream] loading start dataset from: {start_path}")
    dataset = load_dataset(start_path)

    # Filter items
    items: List[DataItem] = list(dataset)
    qid_filter: Optional[Set[int]] = None
    if args.question_id_ranges:
        qid_filter = _parse_question_id_ranges(args.question_id_ranges)
        items = [it for it in items if int(getattr(it, "question_id")) in qid_filter]
        logger.info(f"[stream] filtered to {len(items)} items by question-id-ranges")

    if args.limit is not None:
        items = items[: args.limit]
        logger.info(f"[stream] limit applied: {len(items)} items")

    if not items:
        raise ValueError("No items to process after filtering.")

    # Init stages once
    stages = _StreamStages()
    stages.ensure_value_retrieval_collections({it.database_id for it in items})

    # Streaming outputs
    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    pred_sql_by_qid: Dict[str, str] = {}

    processed = 0
    ex_results: List[int] = []
    gold_failures = 0

    # Decide which remaining stages to run
    remaining = {
        "dataset": ["value_retrieval", "schema_linking", "memory_augmentation", "sql_generation", "sql_revision", "sql_selection"],
        "value_retrieval": ["schema_linking", "memory_augmentation", "sql_generation", "sql_revision", "sql_selection"],
        "schema_linking": ["memory_augmentation", "sql_generation", "sql_revision", "sql_selection"],
        "memory_augmentation": ["sql_generation", "sql_revision", "sql_selection"],
        "sql_generation": ["sql_revision", "sql_selection"],
        "sql_revision": ["sql_selection"],
        "sql_selection": [],
    }[args.start_from]

    # If memory augmentation is disabled by config, skip it even if start_from is earlier
    if not config.memory_augmentation_config.enabled and "memory_augmentation" in remaining:
        remaining = [s for s in remaining if s != "memory_augmentation"]

    try:
        for batch in _iter_batches(items, args.batch_size):
            for data_item in batch:
                processed += 1

                for stage in remaining:
                    if stage == "value_retrieval":
                        stages.run_value_retrieval(data_item)
                    elif stage == "schema_linking":
                        stages.run_schema_linking(data_item)
                    elif stage == "memory_augmentation":
                        stages.run_memory_augmentation(data_item)
                    elif stage == "sql_generation":
                        stages.run_sql_generation(data_item)
                    elif stage == "sql_revision":
                        stages.run_sql_revision(data_item)
                    elif stage == "sql_selection":
                        stages.run_sql_selection(data_item)
                    else:
                        raise ValueError(f"Unknown stage: {stage}")

                # Always compute EA if we have a selected SQL
                pred_sql = getattr(data_item, "final_selected_sql", None)
                if not pred_sql:
                    pred_sql = "Error"
                pred_sql_by_qid[str(data_item.question_id)] = pred_sql

                ex = _eval_ex(pred_sql, data_item.gold_sql, data_item.database_path)
                if ex is None:
                    gold_failures += 1
                else:
                    ex_results.append(ex)

                # Print per-item quick status
                current_ea = (sum(ex_results) / len(ex_results)) if ex_results else 0.0
                _pred_display = "<empty>" if not pred_sql else pred_sql.replace('\n', ' ')
                logger.info(
                    f"[stream] qid={data_item.question_id} done | "
                    f"EA={current_ea:.4f} ({sum(ex_results)}/{len(ex_results)}) | "
                    f"gold_fail={gold_failures} | pred={_pred_display}"
                )

                # Periodic persistence
                if args.save_every and processed % args.save_every == 0:
                    with open(out_json_path, "w", encoding="utf-8") as f:
                        json.dump(pred_sql_by_qid, f, ensure_ascii=False, indent=2)

                    # Persist dataset snapshot to allow resuming from this stage output if desired
                    # We keep saving to the configured selection save path (stream overwrites).
                    try:
                        save_dataset(dataset, config.sql_selection_config.save_path)
                    except Exception as e:
                        logger.warning(f"[stream] failed to save dataset snapshot: {e}")

        # Final save
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(pred_sql_by_qid, f, ensure_ascii=False, indent=2)
        try:
            save_dataset(dataset, config.sql_selection_config.save_path)
        except Exception as e:
            logger.warning(f"[stream] failed to save final dataset: {e}")

        if ex_results:
            ea = sum(ex_results) / len(ex_results)
        else:
            ea = 0.0
        logger.info(
            f"[stream] finished. processed={processed}, evaluated={len(ex_results)}, "
            f"correct={sum(ex_results)}, EA={ea:.4f}, gold_failures={gold_failures}, "
            f"pred_json={out_json_path}"
        )
    finally:
        stages.close()


if __name__ == "__main__":
    main()

