"""Reasoning element evaluator for criminal profiles."""
from typing import Dict, Any
import logging

from ..utils.constants import FRAMEWORKS, PROFILE_SECTIONS
from .base import BaseEvaluator

# Set up logger
logger = logging.getLogger(__name__)

class ReasoningEvaluator(BaseEvaluator):
    """Evaluates the reasoning elements in a profile."""
    
    def __init__(self):
        """Initialize the reasoning evaluator."""
        super().__init__("reasoning")
    
    def evaluate(self, profile: Dict, **kwargs) -> Dict[str, Any]:
        """Evaluate profile reasoning.
        
        Args:
            profile: The profile to evaluate
            
        Returns:
            Dictionary with reasoning counts and details
        """
        section_counts = {}
        total_count = 0
        
        try:
            # Get the offender profile section
            offender_profile = profile.get("offender_profile", {})
            
            # Count overall reasoning
            reasoning = offender_profile.get("reasoning", [])
            if isinstance(reasoning, list):
                total_count += len(reasoning)
                section_counts["overall"] = len(reasoning)
            
            # Check reasoning in each section
            for section in PROFILE_SECTIONS:
                section_data = offender_profile.get(section, {})
                section_reasoning = section_data.get("reasoning", [])
                
                if isinstance(section_reasoning, list):
                    total_count += len(section_reasoning)
                    section_counts[section] = len(section_reasoning)
                else:
                    section_counts[section] = 0
            
            # Framework classifications reasoning
            framework_classifications = profile.get("framework_classifications", {})
            framework_count = 0
            
            for framework in FRAMEWORKS:
                framework_data = framework_classifications.get(framework, {})
                framework_reasoning = framework_data.get("reasoning", [])
                
                if isinstance(framework_reasoning, list):
                    framework_count += len(framework_reasoning)
                    section_counts[f"framework_{framework}"] = len(framework_reasoning)
                else:
                    section_counts[f"framework_{framework}"] = 0
            
            total_count += framework_count
            section_counts["frameworks_total"] = framework_count
            
        except Exception as e:
            logger.error(f"Error evaluating reasoning: {e}")
            return {"reasoning_count": 0, "section_counts": {}}
        
        results = {
            "reasoning_count": total_count,
            "section_counts": section_counts
        }
        
        self.results.append(results)
        return results