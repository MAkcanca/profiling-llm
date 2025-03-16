"""I/O utilities for profile evaluation."""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

def load_profile(profile_path: str) -> Dict:
    """Load a profile from a JSON file.
    
    Args:
        profile_path: Path to the profile JSON file
        
    Returns:
        The profile as a dictionary
    """
    with open(profile_path, "r") as f:
        return json.load(f)

def load_profiles_from_directory(profiles_dir: str) -> List[Dict[str, Any]]:
    """Load all profiles from a directory structure.
    
    Args:
        profiles_dir: Directory containing generated profiles
        
    Returns:
        List of metadata dictionaries for each profile
    """
    profiles_dir = Path(profiles_dir)
    
    # Load metadata file if it exists
    metadata_path = profiles_dir / "generation_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            return metadata.get("metadata", [])
    
    # Scan directory for profile files if no metadata
    profile_metadata = []
    for case_dir in profiles_dir.iterdir():
        if case_dir.is_dir():
            test_case = case_dir.name
            for profile_file in case_dir.glob("*_result.json"):
                model_name = profile_file.stem.replace("_result", "")
                profile_metadata.append({
                    "test_case": test_case,
                    "model": model_name,
                    "result_file": str(profile_file)
                })
    
    return profile_metadata

def load_gold_standards(gold_standards_dir: str) -> Dict[str, str]:
    """Load gold standard profiles for comparison.
    
    Args:
        gold_standards_dir: Directory containing gold standard profiles
        
    Returns:
        Dictionary mapping test case names to gold standard profile paths
    """
    gold_standards_map = {}
    gold_standards_dir = Path(gold_standards_dir)
    
    for gold_file in gold_standards_dir.glob("*.json"):
        test_case = gold_file.stem.replace("-profile", "")
        gold_standards_map[test_case] = str(gold_file)
    
    return gold_standards_map

def create_output_directory(base_dir: str) -> Path:
    """Create a timestamped output directory.
    
    Args:
        base_dir: Base directory for output
        
    Returns:
        Path to the created directory
    """
    # Create base directory if it doesn't exist
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"run_{run_timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    return run_dir