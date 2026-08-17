from .base import BaseChecker
from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from app.prompt import PromptFactory
from typing import Dict, Optional, Tuple
import re


class OrderByLimitChecker(BaseChecker):
    def check_and_revise(self, sql: str, data_item: DataItem, llm: LLM, sampling_budget: int = 1) -> Tuple[str, Dict[str, int]]:
        order_by_limit_suggestion = self._check_order_by_limit(sql)
        if not order_by_limit_suggestion:
            return sql, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.info(f"[OrderByLimitChecker] Found order-by-limit errors in SQL: {sql}")
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
            order_by_limit_suggestion,
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

    def _check_order_by_limit(self, sql: str) -> Optional[str]:
        identifier = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'
        order_by_pattern = re.compile(
            rf"ORDER BY ((MIN|MAX)\(\s*({identifier})\s*\)).*? LIMIT \d+",
            re.IGNORECASE | re.DOTALL,
        )
        res = order_by_pattern.search(sql)
        if not res:
            return None
        return (
            "The SQL uses the ORDER BY function incorrectly, using MIN/MAX in ORDER BY caluse is incrorrect "
            f"(`{res.group()}`), please correct the SQL. If the SQL contains GROUP BY, please judge whether the "
            f"content of `{res.groups()[0]}` needs to use `SUM({res.groups()[2]})`."
        )
