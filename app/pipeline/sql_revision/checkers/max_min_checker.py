from .base import BaseChecker
from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from app.prompt import PromptFactory
from typing import Dict, Optional, Tuple
import re


class MaxMinChecker(BaseChecker):
    def check_and_revise(self, sql: str, data_item: DataItem, llm: LLM, sampling_budget: int = 1) -> Tuple[str, Dict[str, int]]:
        max_min_suggestion = self._check_max_min(sql)
        if not max_min_suggestion:
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.info(f"[MaxMinChecker] Found max-min errors in SQL: {sql}")
        use_memory = PromptFactory.should_use_memory("revision")
        use_caf_mapping = PromptFactory.should_use_context_graph("revision")
        database_schema_profile = PromptFactory.get_enhanced_database_schema_profile(
            data_item,
            use_caf_mapping=use_caf_mapping,
            use_memory=use_memory,
        )
        hint = PromptFactory.get_sql_generation_hint(
            data_item,
            use_caf_mapping=use_caf_mapping,
            use_memory=use_memory,
        )
        prompt = PromptFactory.format_common_checker_prompt(
            database_schema_profile,
            data_item.question,
            hint,
            sql,
            max_min_suggestion,
        )
        parsed_sql_candidate = None
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        while not parsed_sql_candidate and sampling_budget > 0:
            responses, token_usage = llm.ask([{"role": "user", "content": prompt}], n=1, stop=["</result>"])
            response = responses[0].content.strip()
            total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
            total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
            total_token_usage["total_tokens"] += token_usage["total_tokens"]
            try:
                parsed_sql_candidate = self._parse_llm_response(response)
                if parsed_sql_candidate:
                    return parsed_sql_candidate, total_token_usage
            except Exception as e:
                logger.error(f"Error parsing LLM response: {e}")
                logger.debug(f"Response content: {response}")
            sampling_budget -= 1
        return sql, total_token_usage

    def _check_max_min(self, sql: str) -> Optional[str]:
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        max_min_pattern = re.compile(
            rf"=\s*\(\s*SELECT\s*(MAX|MIN)\s*\(\s*({identifier})\s*\)\s*FROM\s*({identifier})",
            re.IGNORECASE | re.DOTALL,
        )
        fun_amb = max_min_pattern.findall(sql)
        order_amb = set(re.findall(r"= (\(SELECT .* LIMIT \d\))", sql, re.IGNORECASE | re.DOTALL))
        select_amb_pattern = re.compile(
            rf"^SELECT[^\(\)]*? ((MIN|MAX)\(\s*{identifier}\s*\)).*?LIMIT 1",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        select_amb = set(select_amb_pattern.findall(sql))

        suggestions = []
        for func_name, col, table in fun_amb:
            order = "DESC" if func_name == "MAX" else "ASC"
            suggestions.append(
                f"WHERE {col} = (SELECT {func_name}({col}) FROM {table}): "
                f"Please use ORDER BY {table}.{col} {order} LIMIT 1 instead of nested SQL"
            )
        for expr in order_amb:
            suggestions.append(f"{expr}: Please use JOIN instead of nested SQL")
        for expr in select_amb:
            suggestions.append(
                f"{expr[0]}: {expr[1]} function is redundant due to LIMIT clause, "
                "please use ORDER BY + LIMIT instead"
            )

        if suggestions:
            return "\n".join(f"{idx + 1}. {suggestion}" for idx, suggestion in enumerate(suggestions))
        return None
