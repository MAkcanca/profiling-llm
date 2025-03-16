#!/usr/bin/env python3
"""
Example script to demonstrate how to evaluate all profile versions using the forensic-llm2 framework.
This script will load profiles from the specified directory, evaluate them using various metrics,
including comparison to gold standards, and produce statistical analysis suitable for a scientific paper.
"""
import os
import sys
import logging
import glob
from pathlib import Path

from forensic_eval.orchestration.orchestrator import EvaluationOrchestrator
from forensic_eval.evaluators.reasoning import ReasoningEvaluator
from forensic_eval.evaluators.framework import FrameworkEvaluator
from forensic_eval.evaluators.gold_standard import GoldStandardEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)

logger = logging.getLogger(__name__)

def find_run_directories(base_dir: str) -> list:
    """Find all run directories in the profiles base directory.
    
    Args:
        base_dir: Base directory containing profile runs
        
    Returns:
        List of paths to run directories
    """
    # Search for run_* directories
    run_dirs = [d for d in glob.glob(f"{base_dir}/run_*") if os.path.isdir(d)]
    
    if not run_dirs:
        logger.warning(f"No run directories found in {base_dir}")
        return []
        
    logger.info(f"Found {len(run_dirs)} run directories: {[os.path.basename(d) for d in run_dirs]}")
    return run_dirs

def main():
    """Run the scientific evaluation of all profile versions."""
    # Set up hardcoded paths based on project structure
    profiles_base_dir = "generated_profiles"
    gold_standards_dir = Path("gold-standards")
    output_dir = Path("paper_results/scientific_analysis")
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Starting evaluation with profiles from {profiles_base_dir}")
    logger.info(f"Using gold standards from {gold_standards_dir}")
    logger.info(f"Results will be saved to {output_dir}")
    
    # Find run directories containing profiles
    run_dirs = find_run_directories(profiles_base_dir)
    if not run_dirs:
        logger.error(f"No run directories found in {profiles_base_dir}. Please check your profile structure.")
        return 1
    
    # Create the evaluation orchestrator
    orchestrator = EvaluationOrchestrator(output_dir=output_dir)
    
    # Add evaluators
    logger.info("Adding evaluators...")
    orchestrator.add_evaluator(ReasoningEvaluator())
    orchestrator.add_evaluator(FrameworkEvaluator())
    
    # Add the gold standard evaluator if gold standards are available
    if gold_standards_dir.exists() and any(gold_standards_dir.glob("*.json")):
        logger.info("Adding Gold Standard Evaluator")
        orchestrator.add_evaluator(GoldStandardEvaluator())
        
        # Load gold standards
        gold_standards_map = {}
        for gold_file in gold_standards_dir.glob("*.json"):
            # Extract the test case name from the filename
            test_case = gold_file.stem
            if test_case.endswith("-profile"):
                test_case = test_case[:-8]  # Remove "-profile" suffix if present
                
            gold_standards_map[test_case] = str(gold_file)
        
        logger.info(f"Loaded {len(gold_standards_map)} gold standards")
        orchestrator.load_gold_standards(gold_standards_map)
    else:
        logger.warning("No gold standards found. Gold standard evaluation will be skipped.")
    
    # Run the evaluation on all profile versions in each run directory
    logger.info("Starting evaluation process...")
    
    try:
        for run_dir in run_dirs:
            logger.info(f"Evaluating profiles in {run_dir}")
            # Set use_all_versions to True to evaluate all profile versions
            results = orchestrator.evaluate_profiles_from_dir(run_dir, use_all_versions=True)
        
        if not orchestrator.results:
            logger.warning("No profiles were evaluated. Check if the profile directories contain valid profiles.")
            return 1
            
        logger.info(f"Evaluation complete. {len(orchestrator.results)} profiles evaluated across {len(run_dirs)} run directories.")
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Key files generated in output directory:")
        logger.info(f"  - evaluation_results.json: Raw evaluation results")
        logger.info(f"  - evaluation_summary.csv: Summary data in CSV format")
        logger.info(f"  - analysis/statistical_analysis.json: Statistical analysis of results")
        logger.info(f"  - analysis/paper_summary.txt: Paper-ready summary of key statistical findings")
        logger.info(f"  - plots/: Various visualizations of evaluation results")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 