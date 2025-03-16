"""Orchestrator for the evaluation process."""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..evaluators.base import BaseEvaluator
from ..utils.io import load_profiles_from_directory, create_output_directory
from ..reporting.export import export_results, create_summary_dataframe
from ..reporting.visualization import generate_all_visualizations

# Set up logger
logger = logging.getLogger(__name__)

class EvaluationOrchestrator:
    """Orchestrates the evaluation process using multiple evaluators."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize the evaluation orchestrator.
        
        Args:
            output_dir: Directory to store evaluation results
        """
        self.output_dir = output_dir
        self.evaluators = {}
        self.results = []
        self.gold_standards = {}
        
        # Create timestamped run directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = create_output_directory(output_dir)
        self.plots_dir = self.run_dir / "plots"
    
    def add_evaluator(self, evaluator: BaseEvaluator) -> None:
        """Add an evaluator to the orchestrator.
        
        Args:
            evaluator: The evaluator to add
        """
        self.evaluators[evaluator.name] = evaluator
        logger.info(f"Added evaluator: {evaluator.name}")
    
    def load_gold_standards(self, gold_standards_map: Dict[str, str]) -> None:
        """Load gold standard profiles for comparison.
        
        Args:
            gold_standards_map: Dictionary mapping test case names to gold standard profile paths
        """
        for test_case, profile_path in gold_standards_map.items():
            try:
                with open(profile_path, "r") as f:
                    self.gold_standards[test_case] = json.load(f)
                logger.info(f"Loaded gold standard for test case: {test_case}")
            except Exception as e:
                logger.error(f"Error loading gold standard {profile_path}: {e}")
    
    def evaluate_profiles_from_dir(self, profiles_dir: str, use_all_versions: bool = True) -> None:
        """Evaluate all profiles in a directory.
        
        Args:
            profiles_dir: Directory containing generated profiles
            use_all_versions: Whether to evaluate all versions of profiles (True) or
                             just the most recent one (False)
        """
        logger.info(f"Evaluating profiles from: {profiles_dir}")
        
        # Load profile metadata
        profile_metadata = load_profiles_from_directory(profiles_dir)
        
        if not profile_metadata:
            logger.error(f"No profiles found in directory: {profiles_dir}")
            return
            
        logger.info(f"Found {len(profile_metadata)} profiles to evaluate")
        
        # Process metadata to handle multiple versions or just the latest
        if not use_all_versions:
            # Filter to only keep the latest version for each test case/model pair
            latest_metadata = {}
            for item in profile_metadata:
                key = (item.get('test_case', ''), item.get('model', ''))
                
                # Skip items without necessary information
                if not all(key):
                    continue
                    
                # Track the latest timestamp
                if key not in latest_metadata or item.get('timestamp', 0) > latest_metadata[key].get('timestamp', 0):
                    latest_metadata[key] = item
            
            # Convert back to list
            profile_metadata = list(latest_metadata.values())
            logger.info(f"Using only the most recent version: {len(profile_metadata)} profiles")
        else:
            logger.info(f"Using all versions: {len(profile_metadata)} total profiles")
            
        # Evaluate each profile
        for item in profile_metadata:
            if "error" in item:
                logger.info(f"Skipping failed profile: {item['test_case']} - {item['model']}")
                continue
                
            if "result_file" not in item:
                logger.info(f"No result file for: {item['test_case']} - {item['model']}")
                continue
                
            profile_path = item["result_file"]
            
            # Include version information in log if available
            version_info = f" (version: {item.get('version', 'unknown')})" if 'version' in item else ""
            logger.info(f"Evaluating profile: {item['test_case']} - {item['model']}{version_info}")
            
            # Load the profile
            try:
                with open(profile_path, "r") as f:
                    profile = json.load(f)
            except Exception as e:
                logger.error(f"Error loading profile {profile_path}: {e}")
                continue
            
            # Run all evaluators
            evaluation_results = {
                "test_case": item["test_case"],
                "model": item["model"],
            }
            
            # Include version information if available
            if "version" in item:
                evaluation_results["version"] = item["version"]
            if "timestamp" in item:
                evaluation_results["timestamp"] = item["timestamp"]
            
            for name, evaluator in self.evaluators.items():
                try:
                    if name == "gold_standard" and item["test_case"] in self.gold_standards:
                        # Run gold standard evaluator with gold standard
                        gold_results = evaluator.evaluate(profile, self.gold_standards[item["test_case"]])
                        evaluation_results[name] = gold_results
                    elif name != "gold_standard":
                        # Run normal evaluator
                        results = evaluator.evaluate(profile)
                        evaluation_results[name] = results
                except Exception as e:
                    logger.error(f"Error running evaluator {name}: {e}")
                    evaluation_results[name] = {"error": str(e)}
            
            # Add processing time if available in metadata
            if "processing_time" in item:
                evaluation_results["processing_time"] = item["processing_time"]
            
            # Store results
            self.results.append(evaluation_results)
        
        # Save results and generate reports
        self._save_and_report_results()
    
    def _save_and_report_results(self) -> None:
        """Save results and generate reports."""
        if not self.results:
            logger.warning("No evaluation results to save")
            return
        
        try:
            # Export results to files
            df = export_results(self.results, self.run_dir)
            
            # Generate visualizations
            #generate_all_visualizations(self.results, self.plots_dir)
            
            logger.info(f"Evaluation results saved to {self.run_dir}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")