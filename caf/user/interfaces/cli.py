# CLI Feedback Interface - Command line user interaction

import logging
from typing import Optional

from .base import BaseFeedbackInterface
from ..types import FeedbackContext, UserFeedback, FeedbackType
from ..exceptions import FeedbackTimeoutError

logger = logging.getLogger(__name__)

class CLIFeedbackInterface(BaseFeedbackInterface):
    """
    CLI Feedback Interface - command line feedback collection
    
    Provides simple command line interface for collecting user feedback.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("CLIFeedbackInterface initialized")
    
    def collect_feedback(self, context: FeedbackContext) -> UserFeedback:
        """Collect feedback through command line interface"""
        try:
            # Display context information
            self._display_context(context)
            
            # Collect feedback based on type
            if context.feedback_type == FeedbackType.SQL_VALIDATION:
                return self._collect_sql_validation(context)
            elif context.feedback_type == FeedbackType.RESULT_QUALITY:
                return self._collect_quality_feedback(context)
            else:
                return self._collect_general_feedback(context)
                
        except KeyboardInterrupt:
            logger.info("User cancelled feedback collection")
            # Return default "incorrect" feedback if user cancels
            return UserFeedback(
                is_correct=False,
                analysis="User cancelled feedback collection",
                suggestion="Manual review required"
            )
    
    def _display_context(self, context: FeedbackContext) -> None:
        """Display feedback context to user"""
        print("\n=== User Feedback Request ===")
        print(f"Original Query: {context.user_query}")
        print(f"Generated SQL: {context.generated_sql}")
        
        if context.execution_result:
            print(f"Execution Result: {context.execution_result}")
        
        # Display prompt message if provided, otherwise use default
        prompt_msg = context.prompt_message or f"Please validate the generated SQL for the question: '{context.user_query}'"
        print(f"\n{prompt_msg}")
        print("-" * 50)
    
    def _collect_sql_validation(self, context: FeedbackContext) -> UserFeedback:
        """Collect SQL validation feedback"""
        print(f"\nGenerated SQL: {context.generated_sql}")
        
        # Collect core feedback
        is_correct = self._get_boolean_input("Is the SQL correct?")
        
        error_category = None
        error_subcategory = None
        suggestion = None
        
        if not is_correct:
            error_category = self._get_text_input("Error category (optional):")
            error_subcategory = self._get_text_input("Error subcategory (optional):")
            suggestion = self._get_text_input("Correction suggestion:")
        
        analysis = self._get_text_input("Additional comments (optional):")
        
        return UserFeedback(
            is_correct=is_correct,
            error_category=error_category,
            error_subcategory=error_subcategory,
            analysis=analysis,
            suggestion=suggestion
        )
    
    def _collect_quality_feedback(self, context: FeedbackContext) -> UserFeedback:
        """Collect result quality feedback"""
        print(f"\nGenerated SQL: {context.generated_sql}")
        
        if context.execution_result:
            print(f"Execution Result: {context.execution_result}")
        
        # Simplified quality assessment
        is_correct = self._get_boolean_input("Is the result satisfactory?")
        analysis = self._get_text_input("Quality assessment:")
        suggestion = self._get_text_input("Improvement suggestions (optional):")
        
        return UserFeedback(
            is_correct=is_correct,
            analysis=analysis,
            suggestion=suggestion
        )
    
    def _collect_general_feedback(self, context: FeedbackContext) -> UserFeedback:
        """Collect general feedback"""
        is_correct = self._get_boolean_input("Is the output correct?")
        analysis = self._get_text_input("Comments:")
        suggestion = self._get_text_input("Suggestions (optional):")
        
        return UserFeedback(
            is_correct=is_correct,
            analysis=analysis,
            suggestion=suggestion
        )
    
    def _get_boolean_input(self, prompt: str) -> bool:
        """Get boolean input from user"""
        while True:
            try:
                response = input(f"{prompt} (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y/yes' or 'n/no'")
            except (EOFError, KeyboardInterrupt):
                raise
    
    def _get_text_input(self, prompt: str) -> Optional[str]:
        """Get text input from user"""
        try:
            response = input(f"{prompt} ").strip()
            return response if response else None
        except (EOFError, KeyboardInterrupt):
            raise
    
    def cleanup(self) -> None:
        """Cleanup CLI interface resources"""
        super().cleanup()
        logger.debug("CLIFeedbackInterface cleanup completed")
