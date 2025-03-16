"""Base evaluator class for profile evaluation."""
import json
import pandas as pd
from typing import Dict, List, Any

class BaseEvaluator:
    """Base class for all evaluators, providing common functionality."""
    
    def __init__(self, name: str):
        """Initialize the evaluator.
        
        Args:
            name: Name of the evaluator
        """
        self.name = name
        self.results = []
    
    def evaluate(self, profile: Dict, **kwargs) -> Dict[str, Any]:
        """Evaluate a profile.
        
        Args:
            profile: The profile to evaluate
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.
        
        Returns:
            DataFrame of evaluation results
        """
        return pd.DataFrame(self.results)
    
    @staticmethod
    def load_profile(profile_path: str) -> Dict:
        """Load a profile from a JSON file.
        
        Args:
            profile_path: Path to the profile JSON file
            
        Returns:
            The profile as a dictionary
        """
        with open(profile_path, "r") as f:
            return json.load(f)