"""Framework classification evaluator for criminal profiles."""
import statistics
import logging
from typing import Dict, Any

from ..utils.constants import FRAMEWORKS, FRAMEWORK_REQUIRED_FIELDS
from .base import BaseEvaluator

# Set up logger
logger = logging.getLogger(__name__)

class FrameworkEvaluator(BaseEvaluator):
    """Evaluates framework classifications in a profile."""
    
    def __init__(self):
        """Initialize the framework evaluator."""
        super().__init__("framework")
    
    def evaluate(self, profile: Dict, **kwargs) -> Dict[str, Any]:
        """Evaluate framework classifications.
        
        Args:
            profile: The profile to evaluate
            
        Returns:
            Dictionary with framework metrics
        """
        framework_scores = {}
        confidence_scores = []
        framework_completeness = {}
        
        try:
            # Determine where framework classifications are stored
            framework_classifications = None
            
            if "framework_classifications" in profile:
                framework_classifications = profile["framework_classifications"]
            elif "k75" in profile:  # Handle BAML schema format
                framework_classifications = profile["k75"]
                
            if not framework_classifications:
                return {
                    "avg_confidence": 0.0,
                    "framework_confidence": {},
                    "framework_completeness": {f: 0.0 for f in FRAMEWORKS}
                }
            
            # Evaluate each framework
            for framework in FRAMEWORKS:
                framework_data = framework_classifications.get(framework, {})
                
                # Check confidence score
                confidence = framework_data.get("confidence")
                if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                    confidence_scores.append(confidence)
                    framework_scores[framework] = confidence
                else:
                    framework_scores[framework] = 0.0
                
                # Check completeness
                filled_fields = sum(1 for field in FRAMEWORK_REQUIRED_FIELDS if framework_data.get(field))
                framework_completeness[framework] = filled_fields / len(FRAMEWORK_REQUIRED_FIELDS)
            
        except Exception as e:
            logger.error(f"Error evaluating frameworks: {e}")
            return {
                "avg_confidence": 0.0,
                "framework_confidence": {},
                "framework_completeness": {f: 0.0 for f in FRAMEWORKS}
            }
        
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        results = {
            "avg_confidence": avg_confidence,
            "framework_confidence": framework_scores,
            "framework_completeness": framework_completeness
        }
        
        self.results.append(results)
        return results