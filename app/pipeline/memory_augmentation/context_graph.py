import json
from pathlib import Path
from typing import Any

from app.config import config
from app.dataset import DataItem
from app.logger import logger

from .base import MemoryAugmentor
from .context_graph_offline_loader import (
    DEFAULT_MAPPING_DECISIONS_PATH,
    ensure_mapping_data,
    format_mapping_historical_qa,
    get_offline_memory_record,
)


class ContextGraphMemoryAugmentor(MemoryAugmentor):
    name = "context_graph"

    @staticmethod
    def _remove_markdown_section(text: str, header_prefix: str) -> str:
        """
        Remove a markdown section that starts with a header line beginning with header_prefix.
        The section is removed up to (but not including) the next '## ' header, or end of text.
        """
        if not text:
            return ""
        lines = text.splitlines()
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(header_prefix):
                i += 1
                while i < len(lines) and not lines[i].startswith("## "):
                    i += 1
                continue
            out.append(line)
            i += 1
        cleaned: list[str] = []
        prev_blank = False
        for l in out:
            blank = l.strip() == ""
            if blank and prev_blank:
                continue
            cleaned.append(l)
            prev_blank = blank
        return "\n".join(cleaned).strip()

    def _get_mapping_path(self) -> str:
        strategy_path = self.strategy_config.get("mapping_decisions_path")
        if strategy_path:
            return strategy_path
        config_path = config.mapping_analysis_config.mapping_decisions_path
        if config_path:
            return config_path
        return str(DEFAULT_MAPPING_DECISIONS_PATH)

    def supports(self, data_item: DataItem) -> bool:
        mapping_path = self._get_mapping_path()
        return Path(mapping_path).exists()

    def augment(self, data_item: DataItem) -> None:
        mapping_path = self._get_mapping_path()

        # Get format_type and include_reasoning from strategy config
        format_type = self.strategy_config.get("format_type", "with_user_interaction")
        include_reasoning = self.strategy_config.get("include_reasoning", False)

        record = get_offline_memory_record(
            data_item.database_id, data_item.question_id, mapping_path
        )
        if format_type == "online_guidance" and record is None:
            raise ValueError(
                "No online guidance record for "
                f"database_id={data_item.database_id}, question_id={data_item.question_id}"
            )

        ensure_mapping_data(data_item, mapping_path, format_type, include_reasoning)

        memory_items = list(getattr(data_item, "memory_items", None) or [])
        historical_qa = list(getattr(data_item, "mapping_historical_qa", None) or [])

        if format_type == "online_guidance":
            guidance_items = record.get("guidance") or []
            for idx, guidance_item in enumerate(guidance_items, start=1):
                if not isinstance(guidance_item, dict):
                    raise ValueError(
                        "Online guidance entries must be objects for "
                        f"question_id={data_item.question_id}"
                    )
                memory_items.append(
                    {
                        "strategy": self.name,
                        "type": "resolved_user_guidance",
                        "rank": idx,
                        "role_id": guidance_item.get("role_id"),
                        "candidate_key": guidance_item.get("candidate_key"),
                        "message": guidance_item.get("message"),
                        "cited_query_ids": guidance_item.get("cited_query_ids") or [],
                    }
                )
            setattr(data_item, "memory_items", memory_items)

            augmented_schema = getattr(data_item, "database_schema_after_mapping", None)
            if augmented_schema:
                setattr(data_item, "memory_schema_overrides", augmented_schema)

            metadata = dict(getattr(data_item, "memory_metadata", None) or {})
            metadata[self.name] = {
                "enabled": True,
                "mapping_path": mapping_path,
                "format_type": format_type,
                "status": record.get("status"),
                "graph_revision": record.get("graph_revision"),
                "n_guidance": len(guidance_items),
                "has_resolved_user_guidance": bool(
                    getattr(data_item, "resolved_user_guidance", None)
                ),
                "has_mapping_schema": bool(augmented_schema),
            }
            setattr(data_item, "memory_metadata", metadata)

            if self.strategy_config.get("log_formatted_blocks"):
                logger.info(
                    f"ContextGraph online guidance | question_id={data_item.question_id} "
                    f"db_id={data_item.database_id} status={record.get('status')}\n"
                    + (getattr(data_item, "resolved_user_guidance", None) or "(no guidance)")
                )
            return

        # Handle wo_user_interaction format
        if format_type == "wo_user_interaction":
            guidance_hint = getattr(data_item, "guidance_hint", None)
            if guidance_hint:
                memory_items.append({
                    "strategy": self.name,
                    "type": "guidance_hint",
                    "format_type": format_type,
                    "content": guidance_hint,
                })
                if self.strategy_config.get("log_formatted_blocks"):
                    log_block = (
                        f"ContextGraph formatted blocks | question_id={data_item.question_id} "
                        f"db_id={data_item.database_id} (wo_user_interaction format)\n"
                        "---------- guidance_hint ----------\n"
                        + guidance_hint
                    )
                    logger.info(log_block)

            setattr(data_item, "memory_items", memory_items)
            metadata = dict(getattr(data_item, "memory_metadata", None) or {})
            metadata[self.name] = {
                "enabled": True,
                "mapping_path": mapping_path,
                "format_type": format_type,
                "has_guidance_hint": bool(guidance_hint),
            }
            setattr(data_item, "memory_metadata", metadata)

            logger.info(
                f"ContextGraphMemoryAugmentor: question_id={getattr(data_item, 'question_id', None)}, "
                f"format_type={format_type}, has_guidance_hint={bool(guidance_hint)}"
            )
            return

        # Existing format: with_user_interaction (default)
        if self.strategy_config.get("log_formatted_blocks"):
            mh = getattr(data_item, "mapping_hint", None) or ""
            hb = format_mapping_historical_qa(historical_qa) if historical_qa else ""
            # Loguru does not expand printf "%s"; avoid "{}" placeholders so hint/SQL text is logged verbatim.
            log_block = (
                f"ContextGraph formatted blocks | question_id={data_item.question_id} "
                f"db_id={data_item.database_id}\n"
                "---------- mapping_hint ----------\n"
                + (mh or "(empty)")
                + "\n---------- historical ----------\n"
                + (hb or "(none)")
            )
            logger.info(log_block)
        for idx, qa in enumerate(historical_qa, start=1):
            memory_items.append(
                {
                    "strategy": self.name,
                    "type": "qa_pair",
                    "rank": idx,
                    "qa_id": qa.get("qa_id"),
                    "question": qa.get("question", ""),
                    "sql": qa.get("sql", ""),
                }
            )

        if record:
            for idx, uc in enumerate(record.get("user_choices") or [], start=1):
                if not isinstance(uc, dict):
                    continue
                mention_type = uc.get("mention_type")
                if mention_type == "structural_join":
                    memory_items.append(
                        {
                            "strategy": self.name,
                            "type": "structural_join",
                            "rank": idx,
                            "mention_text": uc.get("mention_text"),
                            "tables": uc.get("tables"),
                            "no_join": uc.get("no_join", False),
                            "chosen_columns": uc.get("chosen_columns") or [],
                            "chosen_fragment": uc.get("chosen_fragment") or "",
                        }
                    )
                elif mention_type == "semantic_join":
                    memory_items.append(
                        {
                            "strategy": self.name,
                            "type": "semantic_join",
                            "rank": idx,
                            "semantic_tables": uc.get("semantic_tables") or [],
                            "routing_tables": uc.get("routing_tables") or [],
                            "chosen_join_conditions": uc.get("chosen_join_conditions") or [],
                            "no_join": uc.get("no_join", False),
                        }
                    )
                else:
                    frags = list(uc.get("sql_fragments") or [])
                    implicit: list[Any] = []
                    for mn in uc.get("mapping_nodes") or []:
                        if not isinstance(mn, dict):
                            continue
                        frags.extend(mn.get("sql_fragments") or [])
                        implicit.extend(mn.get("implicit_conditions") or [])
                    frags = list(dict.fromkeys(frags))
                    implicit = list(dict.fromkeys(implicit))
                    memory_items.append(
                        {
                            "strategy": self.name,
                            "type": "user_choice",
                            "rank": idx,
                            "mention_text": uc.get("mention_text"),
                            "columns": uc.get("columns"),
                            "sql_fragments": frags,
                            "implicit_conditions": implicit,
                            "mapping_nodes": uc.get("mapping_nodes"),
                        }
                    )
            for idx, g in enumerate(record.get("guidance") or [], start=1):
                if not isinstance(g, dict):
                    continue
                memory_items.append(
                    {
                        "strategy": self.name,
                        "type": "guidance",
                        "rank": idx,
                        "scope": g.get("scope"),
                        "mention_text": g.get("mention_text"),
                        "rules": g.get("rules"),
                    }
                )

        setattr(data_item, "memory_items", memory_items)

        memory_summary = getattr(data_item, "memory_summary", None)
        if memory_summary is not None:
            ms = str(memory_summary).strip()
            ms = self._remove_markdown_section(ms, "## Suggested mappings")
            ms = self._remove_markdown_section(ms, "## Reference: Historical Question-SQL Pairs")
            ms = self._remove_markdown_section(ms, "## User selection")
            ms = self._remove_markdown_section(ms, "## Guidance:")
            setattr(data_item, "memory_summary", ms)

        # Use the augmented schema (schema_linking + missing memory columns) as the
        # schema override. This preserves the full schema_linking context while ensuring
        # all memory-required columns are present — instead of replacing with a highly
        # restricted schema that only contains user_choice columns.
        augmented_schema = getattr(data_item, "database_schema_after_mapping", None)
        if augmented_schema:
            setattr(data_item, "memory_schema_overrides", augmented_schema)

        n_user_choices = len(record.get("user_choices") or []) if record else 0
        n_guidance = len(record.get("guidance") or []) if record else 0
        metadata = dict(getattr(data_item, "memory_metadata", None) or {})
        metadata[self.name] = {
            "enabled": True,
            "mapping_path": mapping_path,
            "n_historical_qa": len(historical_qa),
            "n_user_choices": n_user_choices,
            "n_guidance": n_guidance,
            "has_mapping_schema": bool(augmented_schema),
        }
        setattr(data_item, "memory_metadata", metadata)

        logger.info(
            f"ContextGraphMemoryAugmentor: question_id={getattr(data_item, 'question_id', None)}, "
            f"n_memory_items={len(memory_items)}, n_historical_qa={len(historical_qa)}, "
            f"n_user_choices={n_user_choices}, n_guidance={n_guidance}"
        )
