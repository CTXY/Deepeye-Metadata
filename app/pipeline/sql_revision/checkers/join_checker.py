from .base import BaseChecker
from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from app.prompt import PromptFactory
from typing import Dict, Optional, Tuple
import re


class JoinChecker(BaseChecker):
    def check_and_revise(self, sql: str, data_item: DataItem, llm: LLM, sampling_budget: int = 1) -> Tuple[str, Dict[str, int]]:
        join_suggestion = self._check_join(sql)
        if not join_suggestion:
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.info(f"[JoinChecker] Found join errors in SQL: {sql}")
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
            join_suggestion,
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

    def _check_join(self, sql: str) -> Optional[str]:
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        join_pattern = re.compile(
            rf"JOIN\s+{identifier}(\s+AS\s+{identifier}){{0,1}}\s+ON(\s+{identifier}\.{identifier}\s*(=\s*{identifier}\.{identifier}(?:\s+OR\s+{identifier}\.{identifier}\s*=\s*{identifier}\.{identifier})+|IN\s+\(.*?\)))",
            re.IGNORECASE | re.DOTALL,
        )
        if join_pattern.findall(sql):
            return (
                "The SQL uses the JOIN function incorrectly, due to using `JOIN table AS T ON "
                "Ta.column1 = Tb.column2 OR Ta.column1 = Tb.column3` or "
                "`JOIN table AS T ON Ta.column1 IN`, please only keep the highest priority group "
                "of `Ta.column = Tb.column` in `OR`."
            )
        return None
