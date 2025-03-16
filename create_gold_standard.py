#!/usr/bin/env python3
"""
Create Gold Standard Profiles

This script helps create gold standard profiles from expert input for evaluating
model performance on criminal profiling tasks.

Usage:
    python create_gold_standard.py <test_case_file> <output_file>

Example:
    python create_gold_standard.py test-cases/ted-bundy-lake.md gold-standards/ted-bundy-profile.json
"""

import argparse
import json
import os
from pathlib import Path
from baml_client.types import ProfileResult
from baml_client.types import NarrativeActionSystemType, SexualBehavioralAnalysisType
from baml_client.types import BehavioralChangeStagingType, SpatialBehaviorType

def create_skeleton_profile(case_id="CASE001"):
    """Create an empty profile structure with the correct schema."""
    profile = {
        "case_id": case_id,
        "offender_profile": {
            "demographics": {
                "age_range": "",
                "gender": "",
                "employment": "",
                "relationship": "",
                "living_situation": "",
                "education": "",
                "reasoning": []
            },
            "psychological_characteristics": {
                "personality_type": "",
                "control_needs": "",
                "social_competence": "",
                "stress_factors": "",
                "fantasy_elements": "",
                "anger_management": "",
                "reasoning": []
            },
            "behavioral_characteristics": {
                "crime_scene_behavior": "",
                "violence_type": "",
                "victim_interaction": "",
                "post_offense_behavior": "",
                "risk_taking": "",
                "trophy_taking": "",
                "reasoning": []
            },
            "geographic_behavior": {
                "crime_scene_location": "",
                "crime_scene_layout": "",
                "crime_scene_characteristics": "",
                "reasoning": []
            },
            "skills_and_knowledge": {
                "technical_skills": "",
                "planning_ability": "",
                "weapon_proficiency": "",
                "knot_tying": "",
                "crime_scene_awareness": "",
                "reasoning": []
            },
            "investigative_implications": {
                "prior_offenses": "",
                "escalation_risk": "",
                "documentation": "",
                "victim_selection": "",
                "cooling_off_period": "",
                "reasoning": []
            },
            "key_identifiers": {
                "occupation_type": "",
                "appearance": "",
                "social_role": "",
                "hobbies": "",
                "vehicle": "",
                "reasoning": []
            },
            "reasoning": []
        },
        "validation_metrics": {
            "key_behavioral_indicators": [],
            "critical_evidence": [],
            "profile_accuracy_factors": []
        },
        "framework_classifications": {
            "narrative_action_system": {
                "primary_classification": "",
                "confidence": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasoning": []
            },
            "sexual_behavioral_analysis": {
                "primary_classification": "",
                "confidence": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasoning": []
            },
            "sexual_homicide_pathways_analysis": {
                "primary_classification": "",
                "confidence": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasoning": []
            },
            "spatial_behavioral_analysis": {
                "primary_classification": "",
                "confidence": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasoning": []
            }
        }
    }
    return profile

def extract_case_id(case_file_path):
    """Extract a case ID from the test case filename."""
    filename = Path(case_file_path).stem
    return filename.upper()

def main():
    parser = argparse.ArgumentParser(description="Create gold standard profiles for evaluation")
    parser.add_argument("test_case_file", help="Path to the test case file")
    parser.add_argument("output_file", help="Path to save the gold standard profile")
    args = parser.parse_args()
    
    # Make sure the gold-standards directory exists
    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    
    # Create a skeleton profile with the correct case ID
    case_id = extract_case_id(args.test_case_file)
    profile = create_skeleton_profile(case_id)
    
    # Save the skeleton file for manual editing
    with open(args.output_file, "w") as f:
        json.dump(profile, f, indent=2)
    
    print(f"Skeleton profile created at {args.output_file}")
    print("Please edit this file to create a gold standard profile for evaluation.")
    print("\nFramework options:")
    print("NAS types: VICTIM, PROFESSIONAL, TRAGIC_HERO, REVENGEFUL")
    print("SHPA types: POWER_ASSERTIVE, POWER_REASSURANCE, ANGER_RETALIATORY, ANGER_EXCITATION")
    print("SHPAF types: SADISTIC, ANGRY, OPPORTUNISTIC")
    print("Spatial types: MARAUDER, COMMUTER")

if __name__ == "__main__":
    main() 