"""Gold standard comparison evaluator for criminal profiles."""
import numpy as np
import logging
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer
from ..utils.constants import FRAMEWORKS, PROFILE_SECTIONS
from .base import BaseEvaluator

# Set up logger
logger = logging.getLogger(__name__)

class GoldStandardEvaluator(BaseEvaluator):
    """Evaluates a profile against a gold standard."""
    
    def __init__(self, semantic_model: str = 'all-MiniLM-L6-v2'):
        """Initialize the gold standard evaluator.
        
        Args:
            semantic_model: Name of sentence transformer model for semantic similarity
        """
        super().__init__("gold_standard")
        
        # Load sentence transformer model for semantic similarity
        try:
            self.sentence_model = SentenceTransformer(semantic_model)
            self.semantic_model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load sentence transformer model: {e}")
            logger.warning("Semantic similarity evaluation will be disabled.")
            self.semantic_model_loaded = False
    
    def evaluate(self, profile: Dict, gold_standard: Dict = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a profile against a gold standard.
        
        Args:
            profile: The profile to evaluate
            gold_standard: The gold standard profile
            
        Returns:
            Dictionary with comparison metrics
        """
        metrics = {}
        
        if gold_standard is None:
            logger.warning("No gold standard provided for comparison")
            return {"framework_agreement": 0.0}
        
        try:
            # Framework classification agreement
            profile_classifications = profile.get("framework_classifications", {})
            gold_classifications = gold_standard.get("framework_classifications", {})
            
            metrics["framework_agreement"] = self._evaluate_framework_agreement(
                profile_classifications, gold_classifications)
            
            # Extract framework scores into dedicated accuracy metrics
            framework_scores = metrics["framework_agreement"].get("framework_scores", {})
            framework_matches = metrics["framework_agreement"].get("framework_matches", {})
            
            # Add framework accuracy metrics for visualizations
            metrics["accuracy"] = {}
            metrics["framework_matches"] = {}
            
            # Track which frameworks were applicable (had valid gold standards)
            applicable_frameworks = []
            
            for framework, score in framework_scores.items():
                # Skip frameworks that aren't applicable (null gold standard)
                if score is None:
                    # Still include in matches but mark as not applicable
                    metrics["framework_matches"][framework] = framework_matches[framework]
                    continue
                
                # Only include valid scores in accuracy metrics
                metrics["accuracy"][framework] = score
                applicable_frameworks.append(framework)
                
                # Add detailed match information
                if framework in framework_matches:
                    metrics["framework_matches"][framework] = framework_matches[framework]
                else:
                    metrics["framework_matches"][framework] = {"match": (score == 1.0)}
            
            # Calculate semantic similarity if model is loaded
            if self.semantic_model_loaded:
                metrics["semantic_similarity"] = self._calculate_semantic_similarity(
                    profile, gold_standard)
            
            # Calculate framework contribution metrics
            metrics["framework_contribution"] = self._calculate_framework_contribution(
                applicable_frameworks, framework_scores)
            
            # Log detailed accuracy information for paper
            overall_agreement = metrics["framework_agreement"].get("overall_agreement", 0.0)
            
            # Count only applicable frameworks with non-null scores
            framework_count = len(applicable_frameworks)
            match_count = sum(1 for f in applicable_frameworks if framework_scores.get(f, 0) == 1.0)
            
            logger.info(f"Gold standard comparison - Overall agreement: {overall_agreement:.2f} ({match_count}/{framework_count} applicable frameworks)")
            
            # Log details about inapplicable frameworks
            inapplicable_frameworks = [f for f in framework_scores if framework_scores[f] is None]
            if inapplicable_frameworks:
                logger.info(f"Frameworks not applicable to this case (null gold standard): {', '.join(inapplicable_frameworks)}")
            
            # Log mismatches for applicable frameworks
            for framework, match_info in metrics["framework_matches"].items():
                if isinstance(match_info, dict) and match_info.get("match") is False:  # Explicitly check for False, not None
                    predicted = match_info.get("predicted", "UNKNOWN")
                    gold = match_info.get("gold", "UNKNOWN")
                    logger.info(f"Framework {framework} mismatch: {predicted} != {gold}")
            
            # Log framework contribution information
            if "framework_contribution" in metrics:
                for framework, contribution in metrics["framework_contribution"]["individual_contributions"].items():
                    logger.info(f"Framework {framework} contribution: {contribution:.4f}")
        
        except Exception as e:
            logger.error(f"Error comparing to gold standard: {e}")
            return {"framework_agreement": 0.0}
        
        self.results.append(metrics)
        return metrics
    
    def _evaluate_framework_agreement(self, profile_frameworks: Dict, gold_frameworks: Dict) -> Dict[str, Any]:
        """Evaluate agreement between framework classifications.
        
        Args:
            profile_frameworks: Framework classifications from profile
            gold_frameworks: Framework classifications from gold standard
            
        Returns:
            Dictionary with framework agreement metrics
        """
        framework_match_count = 0
        framework_total = 0
        framework_scores = {}
        framework_matches = {}
        
        for framework in FRAMEWORKS:
            # Check if both profile and gold standard have this framework
            if framework in profile_frameworks and framework in gold_frameworks:
                # Get classification values
                profile_class = profile_frameworks[framework].get("primary_classification")
                gold_class = gold_frameworks[framework].get("primary_classification")
                
                # Skip frameworks where gold standard has null primary_classification
                # This means the framework is not applicable to this case
                if gold_class is None or gold_class == "null" or gold_class == "":
                    logger.info(f"Framework {framework} has null gold standard - skipping from accuracy calculation")
                    framework_scores[framework] = None  # Use None to indicate "not applicable"
                    framework_matches[framework] = {
                        "match": None,  # None means not applicable
                        "predicted": profile_class,
                        "gold": gold_class,
                        "not_applicable": True
                    }
                    continue  # Skip to next framework
                
                # If we have a valid gold standard, include in agreement metrics
                framework_total += 1
                
                if profile_class == gold_class:
                    framework_match_count += 1
                    framework_scores[framework] = 1.0
                    framework_matches[framework] = {
                        "match": True,
                        "predicted": profile_class,
                        "gold": gold_class
                    }
                else:
                    logger.info(f"Framework {framework} mismatch: {profile_class} != {gold_class}")
                    framework_scores[framework] = 0.0
                    framework_matches[framework] = {
                        "match": False,
                        "predicted": profile_class,
                        "gold": gold_class
                    }
            else:
                # If either is missing entirely
                framework_scores[framework] = 0.0
                framework_matches[framework] = {
                    "match": False,
                    "missing_data": True
                }
        
        # Calculate overall agreement only on applicable frameworks
        overall_agreement = framework_match_count / framework_total if framework_total > 0 else 0.0
        
        return {
            "overall_agreement": overall_agreement,
            "framework_scores": framework_scores,
            "framework_matches": framework_matches
        }
    
    def _calculate_framework_contribution(self, applicable_frameworks: List[str], framework_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate contribution metrics for each framework.
        
        Args:
            applicable_frameworks: List of applicable frameworks
            framework_scores: Dictionary of framework scores (1.0 for match, 0.0 for mismatch)
            
        Returns:
            Dictionary with framework contribution metrics
        """
        results = {
            "individual_contributions": {},
            "relative_importance": {},
            "most_influential_frameworks": []
        }
        
        if not applicable_frameworks:
            return results
        
        # Calculate base accuracy with all frameworks
        total_accuracy = sum(framework_scores.get(f, 0) for f in applicable_frameworks) / len(applicable_frameworks)
        
        # For each framework, calculate how much the accuracy would drop if it were wrong
        for framework in applicable_frameworks:
            current_score = framework_scores.get(framework, 0)
            
            if current_score == 1.0:
                # Calculate accuracy if this framework were wrong
                hypothetical_scores = {f: framework_scores.get(f, 0) for f in applicable_frameworks}
                hypothetical_scores[framework] = 0.0
                hypothetical_accuracy = sum(hypothetical_scores.values()) / len(applicable_frameworks)
                
                # Contribution is the drop in accuracy
                contribution = total_accuracy - hypothetical_accuracy
            elif current_score == 0.0:
                # Calculate accuracy if this framework were correct
                hypothetical_scores = {f: framework_scores.get(f, 0) for f in applicable_frameworks}
                hypothetical_scores[framework] = 1.0
                hypothetical_accuracy = sum(hypothetical_scores.values()) / len(applicable_frameworks)
                
                # Potential contribution is the potential gain in accuracy
                contribution = hypothetical_accuracy - total_accuracy
            else:
                contribution = 0.0
            
            results["individual_contributions"][framework] = contribution
        
        # Calculate relative importance of each framework (normalized)
        total_contribution = sum(abs(c) for c in results["individual_contributions"].values())
        if total_contribution > 0:
            for framework, contribution in results["individual_contributions"].items():
                results["relative_importance"][framework] = abs(contribution) / total_contribution
        
        # Identify most influential frameworks (top 3 or all if less than 3)
        sorted_frameworks = sorted(
            applicable_frameworks,
            key=lambda f: abs(results["individual_contributions"].get(f, 0)),
            reverse=True
        )
        results["most_influential_frameworks"] = sorted_frameworks[:min(3, len(sorted_frameworks))]
        
        # Add metadata about the analysis
        results["metadata"] = {
            "total_frameworks": len(applicable_frameworks),
            "total_contribution": total_contribution,
            "baseline_accuracy": total_accuracy
        }
        
        return results
    
    def _calculate_semantic_similarity(self, profile: Dict, gold_standard: Dict) -> Dict[str, Any]:
        """Calculate semantic similarity between profile and gold standard.
        
        Args:
            profile: The profile to evaluate
            gold_standard: The gold standard profile
            
        Returns:
            Dictionary with semantic similarity metrics
        """
        results = {"overall": 0.0, "sections": {}}
        
        # Extract offender profiles
        profile_op = profile.get("offender_profile", {})
        gold_op = gold_standard.get("offender_profile", {})
        
        section_similarities = []
        
        # Calculate similarity for each section
        for section in PROFILE_SECTIONS:
            profile_section = profile_op.get(section, {})
            gold_section = gold_op.get(section, {})
            
            if not profile_section or not gold_section:
                results["sections"][section] = 0.0
                continue
            
            # Convert to text for comparison
            profile_text = self._dict_to_text(profile_section)
            gold_text = self._dict_to_text(gold_section)
            
            # Skip if either text is empty
            if not profile_text or not gold_text:
                results["sections"][section] = 0.0
                continue
                
            # Calculate similarity
            profile_embedding = self.sentence_model.encode(profile_text, show_progress_bar=False)
            gold_embedding = self.sentence_model.encode(gold_text, show_progress_bar=False)
            
            similarity = np.dot(profile_embedding, gold_embedding) / (
                np.linalg.norm(profile_embedding) * np.linalg.norm(gold_embedding)
            )
            
            results["sections"][section] = float(similarity)
            section_similarities.append(float(similarity))
        
        # Overall similarity is average of section similarities
        results["overall"] = np.mean(section_similarities) if section_similarities else 0.0
        
        return results
    
    def _dict_to_text(self, data: Dict) -> str:
        """Convert a dictionary to a text string for semantic comparison.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            Text representation of the dictionary
        """
        text_parts = []
        
        for key, value in data.items():
            if key == "reasoning":
                continue
                
            if isinstance(value, list):
                text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)