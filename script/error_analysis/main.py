"""
Main pipeline for error analysis

Pipeline stages:
1. Load data (NLQ + Incorrect SQL + Correct SQL)
2. Schema checking (filter out pure schema errors)
3. SQL masking (create abstract patterns)
4. Miner Agent (extract error patterns and guidance)
5. Verifier Agent (validate guidance)
6. Save results and insights
"""

import logging
import time
import argparse
from pathlib import Path
from typing import List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from script.error_analysis.config import Config
from script.error_analysis.data_loader import DataLoader
from script.error_analysis.schema_checker import SchemaChecker
from script.error_analysis.sql_masker import SQLMasker
from script.error_analysis.miner_agent import MinerAgent
from script.error_analysis.verifier_agent import VerifierAgent
from script.error_analysis.models import ProcessedSample, Insight
from script.error_analysis import utils

logger = logging.getLogger(__name__)


class ErrorAnalysisPipeline:
    """
    Main pipeline for error analysis
    """
    
    def __init__(self, args):
        self.args = args
        
        # Initialize components
        self.data_loader = DataLoader(
            bird_dev_path=Path(args.bird_dev_json),
            incorrect_results_path=Path(args.incorrect_results_json)
        )
        
        self.schema_checker = SchemaChecker(
            overlap_threshold=args.schema_overlap_threshold
        )
        
        self.sql_masker = SQLMasker()
        
        self.miner_agent = MinerAgent(
            api_key=args.openai_api_key or Config.OPENAI_API_KEY,
            model=args.openai_model,
            temperature=args.openai_temperature
        )
        
        self.verifier_agent = VerifierAgent(
            max_rows=args.max_verification_rows,
            timeout=args.sql_timeout
        )
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'schema_errors_filtered': 0,
            'value_errors_filtered': 0,
            'non_generalizable_filtered': 0,
            'processing_errors': 0,
            'miner_success': 0,
            'miner_failed': 0,
            'verification_passed': 0,
            'verification_failed': 0
        }
    
    def run(self):
        """Run the complete pipeline"""
        logger.info("="*80)
        logger.info("Starting Error Analysis Pipeline")
        logger.info("="*80)
        
        # Ensure output directories exist
        Config.ensure_dirs()
        
        # Stage 1: Load data
        logger.info("\n[Stage 1] Loading data...")
        samples = self.data_loader.load_data()
        self.stats['total_samples'] = len(samples)
        
        if self.args.limit:
            samples = samples[:self.args.limit]
            logger.info(f"Limited to first {self.args.limit} samples")
        
        # Stage 2-5: Process each sample
        processed_samples = []
        
        for idx, sample in enumerate(samples):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing sample {idx+1}/{len(samples)} - Question ID: {sample.question_id}")
            logger.info(f"{'='*80}")
            
            try:
                processed_sample = self._process_sample(sample)
                processed_samples.append(processed_sample)
                
                # Save intermediate result
                if self.args.save_intermediate:
                    self._save_intermediate_sample(processed_sample)
                
                # Rate limiting
                if idx < len(samples) - 1:
                    time.sleep(self.args.delay)
                    
            except Exception as e:
                logger.error(f"Failed to process sample {sample.question_id}: {e}")
                sample.processing_status = "failed"
                sample.error_message = str(e)
                processed_samples.append(sample)
                self.stats['processing_errors'] += 1
        
        # Stage 6: Generate insights and save results
        logger.info("\n[Stage 6] Generating insights...")
        self._generate_and_save_insights(processed_samples)
        
        # Print summary
        self._print_summary()
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline completed!")
        logger.info("="*80)
    
    def _process_sample(self, sample: ProcessedSample) -> ProcessedSample:
        """Process a single sample through all stages"""
        
        # Stage 2: Schema checking
        logger.info(f"[Stage 2] Checking schema...")
        (
            is_pure_schema_error,
            overlap_score,
            incorrect_schema,
            correct_schema,
            qualified_incorrect_sql,
            qualified_correct_sql
        ) = self.schema_checker.check(
            sample.incorrect_sql,
            sample.correct_sql,
            sample.db_path
        )
        
        sample.is_pure_schema_error = is_pure_schema_error
        sample.schema_overlap_score = overlap_score
        sample.incorrect_schema = incorrect_schema
        sample.correct_schema = correct_schema
        sample.qualified_incorrect_sql = qualified_incorrect_sql
        sample.qualified_correct_sql = qualified_correct_sql
        
        if is_pure_schema_error:
            logger.info(f"✗ Filtered out as PURE SCHEMA ERROR (overlap={overlap_score:.3f})")
            sample.processing_status = "schema_error_filtered"
            self.stats['schema_errors_filtered'] += 1
            return sample
        
        logger.info(f"✓ Passed schema check (overlap={overlap_score:.3f})")
        
        # Stage 3: SQL masking
        logger.info(f"[Stage 3] Masking SQLs...")
        (
            masked_incorrect_sql,
            masked_correct_sql,
            mapping_dict
        ) = self.sql_masker.mask_sql_pair(
            qualified_incorrect_sql,
            qualified_correct_sql
        )
        
        sample.masked_incorrect_sql = masked_incorrect_sql
        sample.masked_correct_sql = masked_correct_sql
        sample.mapping_dict = mapping_dict
        
        logger.info(f"✓ Masked incorrect: {masked_incorrect_sql[:100]}...")
        logger.info(f"✓ Masked correct: {masked_correct_sql[:100]}...")
        
        # Stage 3.5: Check for value-only differences
        is_value_only_error = self.sql_masker.check_value_only_difference(
            masked_incorrect_sql,
            masked_correct_sql
        )
        
        if is_value_only_error:
            logger.info(f"✗ Filtered out as VALUE-ONLY ERROR (only literal values differ)")
            sample.processing_status = "value_error_filtered"
            self.stats['value_errors_filtered'] += 1
            return sample
        
        logger.info(f"✓ Passed value-only check (structural differences exist)")
        
        # Stage 4: Miner Agent
        logger.info(f"[Stage 4] Running Miner Agent...")
        miner_output = self.miner_agent.analyze(
            nlq=sample.nlq,
            masked_incorrect_sql=masked_incorrect_sql,
            masked_correct_sql=masked_correct_sql,
            evidence=sample.evidence or ""
        )
        
        if miner_output is None:
            logger.warning("✗ Miner Agent failed")
            sample.processing_status = "miner_failed"
            self.stats['miner_failed'] += 1
            return sample
        
        sample.miner_output = miner_output
        self.stats['miner_success'] += 1
        logger.info(f"✓ Miner Agent succeeded")
        logger.info(f"  Intent: {miner_output.guidance.intent[:80]}...")
        logger.info(f"  NL Triggers: {miner_output.retrieval_key.nl_triggers}")
        logger.info(f"  SQL Risk Atoms: {miner_output.retrieval_key.sql_risk_atoms}")
        logger.info(f"  Generalizable: {miner_output.is_generalizable}")
        
        # Check generalizability
        if not miner_output.is_generalizable:
            logger.info(f"✗ Filtered out as NON-GENERALIZABLE insight")
            logger.info(f"  Reason: {miner_output.generalizability_reason}")
            sample.processing_status = "non_generalizable_filtered"
            self.stats['non_generalizable_filtered'] += 1
            return sample
        
        # Stage 5: Verifier Agent
        logger.info(f"[Stage 5] Running Verifier Agent...")
        verification_passed, verification_details = self.verifier_agent.verify(
            incorrect_sql=sample.incorrect_sql,
            correct_sql=sample.correct_sql,
            db_path=sample.db_path,
            miner_output=miner_output
        )
        
        sample.verification_passed = verification_passed
        sample.verification_details = verification_details
        
        if verification_passed:
            logger.info(f"✓ Verification PASSED")
            self.stats['verification_passed'] += 1
            sample.processing_status = "verified_success"
        else:
            logger.info(f"✗ Verification FAILED")
            self.stats['verification_failed'] += 1
            sample.processing_status = "verified_failed"
        
        logger.info(f"  {verification_details.get('execution_summary', '')}")
        
        return sample
    
    def _save_intermediate_sample(self, sample: ProcessedSample):
        """Save intermediate result for a single sample"""
        output_file = Config.INTERMEDIATE_DIR / f"sample_{sample.question_id}.json"
        utils.save_json(sample, output_file)
    
    def _generate_and_save_insights(self, processed_samples: List[ProcessedSample]):
        """Generate insights from processed samples and save"""
        
        # Filter samples that successfully generated insights
        successful_samples = [
            s for s in processed_samples 
            if s.miner_output is not None and s.processing_status in ["verified_success", "verified_failed"]
        ]
        
        logger.info(f"Generating insights from {len(successful_samples)} successful samples")
        
        # For now, create one insight per sample
        # TODO: In future, cluster similar patterns and merge insights
        insights = []
        
        for sample in successful_samples:
            insight = Insight(
                insight_id=f"insight_{sample.question_id}",
                retrieval_key=sample.miner_output.retrieval_key,
                guidance=sample.miner_output.guidance,
                qualified_incorrect_sql=sample.qualified_incorrect_sql,
                qualified_correct_sql=sample.qualified_correct_sql,
                source_question_ids=[sample.question_id],
                verification_success_count=1 if sample.verification_passed else 0,
                verification_total_count=1,
                verification_success_rate=1.0 if sample.verification_passed else 0.0,
                created_at=utils.get_timestamp()
            )
            insights.append(insight)
        
        # Save insights
        if insights:
            utils.save_jsonl(insights, Config.INSIGHTS_FILE)
            logger.info(f"✓ Saved {len(insights)} insights to {Config.INSIGHTS_FILE}")
        else:
            logger.warning("No insights generated")
        
        # Save all processed samples
        all_samples_file = Config.OUTPUT_DIR / "all_processed_samples.jsonl"
        utils.save_jsonl(processed_samples, all_samples_file)
        logger.info(f"✓ Saved all processed samples to {all_samples_file}")
    
    def _print_summary(self):
        """Print pipeline statistics"""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE STATISTICS")
        logger.info("="*80)
        logger.info(f"Total samples:              {self.stats['total_samples']}")
        logger.info(f"Schema errors filtered:     {self.stats['schema_errors_filtered']}")
        logger.info(f"Value errors filtered:      {self.stats['value_errors_filtered']}")
        logger.info(f"Non-generalizable filtered: {self.stats.get('non_generalizable_filtered', 0)}")
        logger.info(f"Processing errors:          {self.stats['processing_errors']}")
        logger.info(f"Miner success:              {self.stats['miner_success']}")
        logger.info(f"Miner failed:               {self.stats['miner_failed']}")
        logger.info(f"Verification passed:        {self.stats['verification_passed']}")
        logger.info(f"Verification failed:        {self.stats['verification_failed']}")
        logger.info("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Error Analysis Pipeline for SQL Generation"
    )
    
    # Input files
    parser.add_argument(
        '--bird-dev-json',
        type=str,
        default=str(Config.BIRD_DEV_JSON),
        help='Path to bird dev.json file'
    )
    parser.add_argument(
        '--incorrect-results-json',
        type=str,
        default=str(Config.INCORRECT_RESULTS_JSON),
        help='Path to incorrect results JSON file'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Config.OUTPUT_DIR),
        help='Output directory'
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        default=True,
        help='Save intermediate results for each sample'
    )
    
    # Processing options
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples to process (for testing)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=Config.DELAY_BETWEEN_REQUESTS,
        help='Delay between API requests (seconds)'
    )
    
    # Schema checker
    parser.add_argument(
        '--schema-overlap-threshold',
        type=float,
        default=Config.SCHEMA_OVERLAP_THRESHOLD,
        help='Schema overlap threshold for filtering'
    )
    
    # LLM configuration
    parser.add_argument(
        '--openai-api-key',
        type=str,
        default=None,
        help='OpenAI API key (overrides env variable)'
    )
    parser.add_argument(
        '--openai-model',
        type=str,
        default=Config.OPENAI_MODEL,
        help='OpenAI model name'
    )
    parser.add_argument(
        '--openai-temperature',
        type=float,
        default=Config.OPENAI_TEMPERATURE,
        help='OpenAI temperature'
    )
    
    # Verifier configuration
    parser.add_argument(
        '--max-verification-rows',
        type=int,
        default=100,
        help='Maximum rows to fetch for verification'
    )
    parser.add_argument(
        '--sql-timeout',
        type=int,
        default=30,
        help='SQL execution timeout (seconds)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    utils.setup_logging(args.log_level, log_file)
    
    # Update config from args
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.INTERMEDIATE_DIR = Config.OUTPUT_DIR / "intermediate"
    Config.INSIGHTS_FILE = Config.OUTPUT_DIR / "insights.jsonl"
    
    # Run pipeline
    pipeline = ErrorAnalysisPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

