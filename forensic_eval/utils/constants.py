"""Constants used throughout the forensic evaluation package."""

# Supported frameworks for evaluation
FRAMEWORKS = [
    "narrative_action_system",
    "sexual_behavioral_analysis",
    "spatial_behavioral_analysis",
    "sexual_homicide_pathways_analysis"
]

# Profile sections for evaluation
PROFILE_SECTIONS = [
    "demographics",
    "psychological_characteristics",
    "behavioral_characteristics",
    "geographic_behavior",
    "skills_and_knowledge",
    "investigative_implications",
    "key_identifiers"
]

# Required fields for framework evaluation
FRAMEWORK_REQUIRED_FIELDS = [
    "primary_classification", 
    "confidence", 
    "supporting_evidence", 
    "contradicting_evidence",
    "reasoning"
]