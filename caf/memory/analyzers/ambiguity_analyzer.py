"""
Ambiguity Analyzer - 统一的模糊字段分析器

协调整个分析流程：
1. 从 miners 收集 ambiguous pairs
2. 去重合并
3. 统计待分析数量
4. 多线程分析 (数据内容 + 语义意图)
5. 生成完整的 DiffProfile
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from ..types import AmbiguousPair, DiffProfile, DataContentProfile, SemanticIntentProfile
from ..stores.ambiguous_pair import AmbiguousPairStore
from ..generators.pseudo_query_collision_miner import PseudoQueryCollisionMiner
from ..generators.value_overlap_cluster_miner import ValueOverlapClusterMiner
from .data_content_analyzer import DataContentAnalyzer
from .semantic_intent_analyzer import SemanticIntentAnalyzer

logger = logging.getLogger(__name__)


class AmbiguityAnalyzer:
    """
    统一的模糊字段分析器.
    
    工作流程：
    1. Mining Stage: 调用 miners 挖掘 ambiguous pairs
    2. Deduplication: 去重合并来自不同 miners 的 pairs
    3. Analysis Stage: 分析每个 pair 的 DiffProfile
       - 数据内容维度 (DataContentAnalyzer)
       - 语义意图维度 (SemanticIntentAnalyzer)
    4. Save Results: 保存完整的分析结果
    """

    def __init__(
        self,
        pair_store: AmbiguousPairStore,
        pseudo_query_miner: PseudoQueryCollisionMiner,
        value_overlap_miner: ValueOverlapClusterMiner,
        data_content_analyzer: DataContentAnalyzer,
        semantic_intent_analyzer: SemanticIntentAnalyzer,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            pair_store: Store for ambiguous pairs
            pseudo_query_miner: Miner for pseudo-query collisions
            value_overlap_miner: Miner for value overlap
            data_content_analyzer: Analyzer for data content dimension
            semantic_intent_analyzer: Analyzer for semantic intent dimension
            config: Configuration dict
        """
        self.pair_store = pair_store
        self.pseudo_query_miner = pseudo_query_miner
        self.value_overlap_miner = value_overlap_miner
        self.data_content_analyzer = data_content_analyzer
        self.semantic_intent_analyzer = semantic_intent_analyzer
        self.config = config or {}

        # Configuration
        self.num_workers = self.config.get("num_workers", 4)  # Number of parallel threads
        self.enable_data_content = self.config.get("enable_data_content", True)
        self.enable_semantic_intent = self.config.get("enable_semantic_intent", True)

    def analyze_database(self, database_id: str) -> Dict[str, any]:
        """
        完整的分析流程：挖掘 -> 去重 -> 分析 -> 保存.
        
        Args:
            database_id: Database identifier
        
        Returns:
            Summary statistics
        """
        logger.info("="*80)
        logger.info("Starting ambiguity analysis for database: %s", database_id)
        logger.info("="*80)

        # Pre-bind stores to avoid repeated binding during parallel analysis
        logger.info("\nPre-binding stores for database: %s", database_id)
        self.pair_store.bind_database(database_id)
        
        # Pre-bind semantic store if available (for semantic intent analyzer)
        if hasattr(self.semantic_intent_analyzer, 'semantic_store') and \
           self.semantic_intent_analyzer.semantic_store:
            self.semantic_intent_analyzer.semantic_store.bind_database(database_id)
            logger.info("Pre-bound semantic store for metadata access")

        # Step 1: Mining
        logger.info("\n[Step 1/4] Mining ambiguous pairs...")
        pairs = self._mine_ambiguous_pairs(database_id)

        if not pairs:
            logger.info("No ambiguous pairs found. Analysis complete.")
            return {
                "database_id": database_id,
                "total_pairs": 0,
                "analyzed_pairs": 0,
                "status": "no_pairs_found",
            }

        # Step 2: Deduplication and merge
        logger.info("\n[Step 2/4] Deduplicating and merging pairs...")
        unique_pairs = self._deduplicate_and_merge_pairs(pairs)

        # Save merged pairs (before analysis) - already bound, no need to bind again
        self.pair_store.save_pairs(database_id, unique_pairs)

        logger.info("After deduplication: %d unique pairs", len(unique_pairs))

        # Step 3: Analysis
        logger.info("\n[Step 3/4] Analyzing %d pairs...", len(unique_pairs))
        analyzed_pairs = self._analyze_pairs_parallel(unique_pairs, database_id)

        # Step 4: Save results
        logger.info("\n[Step 4/4] Saving analysis results...")
        self.pair_store.save_pairs(database_id, analyzed_pairs)

        # Summary statistics
        stats = self._compute_statistics(analyzed_pairs)
        stats["database_id"] = database_id

        logger.info("\n" + "="*80)
        logger.info("Ambiguity analysis completed for database: %s", database_id)
        logger.info("Total pairs: %d", stats["total_pairs"])
        logger.info("Successfully analyzed: %d", stats["analyzed_pairs"])
        logger.info("Failed: %d", stats["failed_pairs"])
        logger.info("="*80 + "\n")

        return stats

    def _mine_ambiguous_pairs(self, database_id: str) -> List[AmbiguousPair]:
        """
        Step 1: 调用两个 miners 挖掘 ambiguous pairs.
        """
        all_pairs: List[AmbiguousPair] = []

        # Mine pseudo-query collisions
        try:
            logger.info("Mining pseudo-query collisions...")
            pseudo_pairs = self.pseudo_query_miner.mine_and_save_pairs(database_id)
            logger.info("Found %d pairs from pseudo-query collision", len(pseudo_pairs))
            all_pairs.extend(pseudo_pairs)
        except Exception as e:
            logger.error("Failed to mine pseudo-query collisions: %s", e)

        # Mine value overlap
        try:
            logger.info("Mining value overlap...")
            overlap_pairs = self.value_overlap_miner.mine_and_save_pairs(database_id)
            logger.info("Found %d pairs from value overlap", len(overlap_pairs))
            all_pairs.extend(overlap_pairs)
        except Exception as e:
            logger.error("Failed to mine value overlap: %s", e)

        logger.info("Total pairs from both miners: %d", len(all_pairs))
        return all_pairs

    def _deduplicate_and_merge_pairs(
        self, pairs: List[AmbiguousPair]
    ) -> List[AmbiguousPair]:
        """
        Step 2: 去重并合并来自不同 miners 的 pairs.
        
        合并策略：
        - 相同的 (column_a, column_b) pair 合并为一个
        - 合并 discovery_methods
        - 保留两个 miners 的所有信号 (semantic_collision_score, value_jaccard)
        """
        if not pairs:
            return []

        # Group by canonical pair key
        pair_groups: Dict[Tuple[Tuple[str, str], Tuple[str, str]], List[AmbiguousPair]] = {}

        for pair in pairs:
            pair_key = pair.get_sorted_pair_key()
            if pair_key not in pair_groups:
                pair_groups[pair_key] = []
            pair_groups[pair_key].append(pair)

        logger.info(
            "Deduplication: %d raw pairs -> %d unique pairs",
            len(pairs),
            len(pair_groups),
        )

        # Merge duplicates
        merged_pairs: List[AmbiguousPair] = []
        pair_idx = 0

        for pair_key, group in pair_groups.items():
            if len(group) == 1:
                # No duplicates
                merged_pairs.append(group[0])
            else:
                # Merge multiple pairs
                merged_pair = self._merge_pair_group(group, pair_idx)
                merged_pairs.append(merged_pair)
                pair_idx += 1

        return merged_pairs

    def _merge_pair_group(
        self, group: List[AmbiguousPair], pair_idx: int
    ) -> AmbiguousPair:
        """
        合并一组重复的 pairs.
        """
        # Use first pair as base
        base_pair = group[0]

        # Merge discovery methods
        all_methods: Set[str] = set()
        for pair in group:
            all_methods.update(pair.discovery_methods or [])

        # Take max scores
        semantic_scores = [
            p.semantic_collision_score for p in group if p.semantic_collision_score is not None
        ]
        value_jaccards = [
            p.value_jaccard for p in group if p.value_jaccard is not None
        ]

        semantic_score = max(semantic_scores) if semantic_scores else None
        value_jaccard = max(value_jaccards) if value_jaccards else None

        # Merge collision details
        all_collision_details = []
        for pair in group:
            if pair.collision_details:
                all_collision_details.extend(pair.collision_details)

        # Create merged pair
        merged_pair = AmbiguousPair(
            pair_id=f"{base_pair.database_id}_merged_pair_{pair_idx:04d}",
            database_id=base_pair.database_id,
            column_a=base_pair.column_a,
            column_b=base_pair.column_b,
            discovery_methods=list(all_methods),
            semantic_collision_score=semantic_score,
            value_jaccard=value_jaccard,
            collision_details=all_collision_details if all_collision_details else None,
            diff_profile=None,  # Will be filled in analysis stage
            discovered_at=base_pair.discovered_at,
        )

        logger.debug(
            "Merged %d duplicate pairs into: %s (methods: %s)",
            len(group),
            merged_pair.pair_id,
            all_methods,
        )

        return merged_pair

    def _analyze_pairs_parallel(
        self, pairs: List[AmbiguousPair], database_id: str
    ) -> List[AmbiguousPair]:
        """
        Step 3: 多线程分析 pairs.
        
        对每个 pair 分析：
        - 数据内容维度 (可能较慢，需要 SQL)
        - 语义意图维度 (需要 LLM，可并行)
        """
        analyzed_pairs: List[AmbiguousPair] = []

        logger.info("Analyzing %d pairs using %d workers...", len(pairs), self.num_workers)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(self._analyze_single_pair, pair, database_id): pair
                for pair in pairs
            }

            # Collect results with progress tracking
            completed = 0
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                completed += 1

                try:
                    analyzed_pair = future.result()
                    analyzed_pairs.append(analyzed_pair)

                    # Log progress
                    if completed % 10 == 0 or completed == len(pairs):
                        logger.info(
                            "Progress: %d/%d pairs analyzed (%.1f%%)",
                            completed,
                            len(pairs),
                            100.0 * completed / len(pairs),
                        )

                except Exception as e:
                    logger.error(
                        "Failed to analyze pair %s: %s",
                        pair.pair_id,
                        e,
                    )
                    # Still add the pair, but without DiffProfile
                    analyzed_pairs.append(pair)

        return analyzed_pairs

    def _analyze_single_pair(
        self, pair: AmbiguousPair, database_id: str
    ) -> AmbiguousPair:
        """
        分析单个 pair，生成完整的 DiffProfile.
        """
        logger.debug("Analyzing pair: %s", pair.pair_id)

        data_content_profile: Optional[DataContentProfile] = None
        semantic_intent_profile: Optional[SemanticIntentProfile] = None

        # Data content analysis (optional, can be slow)
        if self.enable_data_content:
            try:
                data_content_profile = self.data_content_analyzer.analyze_pair(pair)
            except Exception as e:
                logger.warning(
                    "Data content analysis failed for %s: %s",
                    pair.pair_id,
                    e,
                )

        # Semantic intent analysis (LLM-based)
        if self.enable_semantic_intent:
            try:
                semantic_intent_profile = self.semantic_intent_analyzer.analyze_pair(
                    pair, database_id
                )
            except Exception as e:
                logger.warning(
                    "Semantic intent analysis failed for %s: %s",
                    pair.pair_id,
                    e,
                )

        # Generate guidance rule
        guidance_rule = self._generate_guidance_rule(
            pair, data_content_profile, semantic_intent_profile
        )

        # Build DiffProfile
        diff_profile = DiffProfile(
            data_content_profile=data_content_profile,
            semantic_intent_profile=semantic_intent_profile,
            guidance_rule=guidance_rule,
            analysis_timestamp=datetime.utcnow().isoformat(),
            analysis_version="v1.0",
        )

        # Update pair
        pair.diff_profile = diff_profile
        pair.last_analyzed_at = datetime.utcnow().isoformat()

        return pair

    def _generate_guidance_rule(
        self,
        pair: AmbiguousPair,
        data_content: Optional[DataContentProfile],
        semantic_intent: Optional[SemanticIntentProfile],
    ) -> Optional[str]:
        """
        综合两个维度，生成最终的 guidance rule.
        
        Guidance rule 是一句话总结，用于提示 Agent 如何选择字段。
        """
        rules = []

        # From data content
        if data_content:
            if data_content.set_relationship == "A_subset_of_B":
                rules.append("Column A is a subset of Column B (more specific/restrictive)")
            elif data_content.set_relationship == "B_subset_of_A":
                rules.append("Column B is a subset of Column A (more specific/restrictive)")

            if data_content.constraint_rule:
                rules.append(f"Logical constraint: {data_content.constraint_rule}")

            if data_content.sensitivity_type == "high_sensitivity":
                rules.append("HIGH RISK: Swapping these columns will return completely different results")
            elif data_content.sensitivity_type == "low_sensitivity":
                rules.append("Low risk: These columns are highly redundant (synonyms)")

        # From semantic intent
        if semantic_intent:
            if semantic_intent.discriminative_logic:
                rules.append(f"Key distinction: {semantic_intent.discriminative_logic}")

            if semantic_intent.trigger_keywords_a and semantic_intent.trigger_keywords_b:
                keywords_a = ", ".join(semantic_intent.trigger_keywords_a[:3])
                keywords_b = ", ".join(semantic_intent.trigger_keywords_b[:3])
                rules.append(
                    f"Use Column A for: [{keywords_a}]; "
                    f"Use Column B for: [{keywords_b}]"
                )

        if not rules:
            return None

        # Combine rules
        guidance = " | ".join(rules)
        return guidance

    def _compute_statistics(self, pairs: List[AmbiguousPair]) -> Dict[str, any]:
        """
        计算统计信息.
        """
        total = len(pairs)
        analyzed = sum(1 for p in pairs if p.diff_profile is not None)
        failed = total - analyzed

        # Count by discovery method
        method_counts: Dict[str, int] = {}
        for pair in pairs:
            for method in pair.discovery_methods or []:
                method_counts[method] = method_counts.get(method, 0) + 1

        return {
            "total_pairs": total,
            "analyzed_pairs": analyzed,
            "failed_pairs": failed,
            "discovery_method_counts": method_counts,
        }












