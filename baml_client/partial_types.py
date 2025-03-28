###############################################################################
#
#  Welcome to Baml! To use this generated code, please run the following:
#
#  $ pip install baml-py
#
###############################################################################

# This file was generated by BAML: please do not edit it. Instead, edit the
# BAML files and re-generate this code.
#
# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off
import baml_py
from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Dict, Generic, List, Optional, TypeVar, Union, Literal

from . import types
from .types import Checked, Check

###############################################################################
#
#  These types are used for streaming, for when an instance of a type
#  is still being built up and any of its fields is not yet fully available.
#
###############################################################################

T = TypeVar('T')
class StreamState(BaseModel, Generic[T]):
    value: T
    state: Literal["Pending", "Incomplete", "Complete"]


class BehavioralCharacteristics(BaseModel):
    crime_scene_behavior: Optional[str] = None
    violence_type: Optional[str] = None
    victim_interaction: Optional[str] = None
    post_offense_behavior: Optional[str] = None
    risk_taking: Optional[str] = None
    trophy_taking: Optional[str] = None
    reasoning: List[str]

class Demographics(BaseModel):
    age_range: Optional[str] = None
    gender: Optional[types.Gender] = None
    employment: Optional[str] = None
    relationship: Optional[str] = None
    living_situation: Optional[str] = None
    education: Optional[str] = None
    reasoning: List[str]

class FrameworkClassifications(BaseModel):
    narrative_action_system: Optional["NarrativeActionSystemClassification"] = None
    sexual_behavioral_analysis: Optional["SexualBehavioralAnalysisClassification"] = None
    sexual_homicide_pathways_analysis: Optional["SexualHomicidePathwaysClassification"] = None
    spatial_behavioral_analysis: Optional["SpatialBehaviorClassification"] = None

class GeographicBehavior(BaseModel):
    crime_scene_location: Optional[str] = None
    crime_scene_layout: Optional[str] = None
    crime_scene_characteristics: Optional[str] = None
    reasoning: List[str]

class InvestigativeImplications(BaseModel):
    prior_offenses: Optional[str] = None
    escalation_risk: Optional[str] = None
    documentation: Optional[str] = None
    victim_selection: Optional[str] = None
    cooling_off_period: Optional[str] = None
    reasoning: List[str]

class KeyIdentifiers(BaseModel):
    occupation_type: Optional[str] = None
    appearance: Optional[str] = None
    social_role: Optional[str] = None
    hobbies: Optional[str] = None
    vehicle: Optional[str] = None
    reasoning: List[str]

class NarrativeActionSystemClassification(BaseModel):
    primary_classification: Optional[types.NarrativeActionSystemType] = None
    confidence: Optional[float] = None
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    reasoning: List[str]

class OffenderProfile(BaseModel):
    demographics: Optional["Demographics"] = None
    psychological_characteristics: Optional["PsychologicalCharacteristics"] = None
    behavioral_characteristics: Optional["BehavioralCharacteristics"] = None
    geographic_behavior: Optional["GeographicBehavior"] = None
    skills_and_knowledge: Optional["SkillsAndKnowledge"] = None
    investigative_implications: Optional["InvestigativeImplications"] = None
    key_identifiers: Optional["KeyIdentifiers"] = None
    reasoning: List[str]

class ProfileResult(BaseModel):
    case_id: Optional[str] = None
    offender_profile: Optional["OffenderProfile"] = None
    validation_metrics: Optional["ValidationMetrics"] = None
    framework_classifications: Optional["FrameworkClassifications"] = None

class PsychologicalCharacteristics(BaseModel):
    personality_type: Optional[str] = None
    control_needs: Optional[str] = None
    social_competence: Optional[str] = None
    stress_factors: Optional[str] = None
    fantasy_elements: Optional[str] = None
    anger_management: Optional[str] = None
    reasoning: List[str]

class SexualBehavioralAnalysisClassification(BaseModel):
    primary_classification: Optional[types.SexualBehavioralAnalysisType] = None
    confidence: Optional[float] = None
    supporting_evidence: Optional[List[str]] = None
    contradicting_evidence: Optional[List[str]] = None
    reasoning: Optional[List[str]] = None

class SexualHomicidePathwaysClassification(BaseModel):
    primary_classification: Optional[types.SexualHomicidePathwaysType] = None
    confidence: Optional[float] = None
    supporting_evidence: Optional[List[str]] = None
    contradicting_evidence: Optional[List[str]] = None
    reasoning: Optional[List[str]] = None

class SkillsAndKnowledge(BaseModel):
    technical_skills: Optional[str] = None
    planning_ability: Optional[str] = None
    weapon_proficiency: Optional[str] = None
    knot_tying: Optional[str] = None
    crime_scene_awareness: Optional[str] = None
    reasoning: List[str]

class SpatialBehaviorClassification(BaseModel):
    primary_classification: Optional[types.SpatialBehaviorType] = None
    confidence: Optional[float] = None
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    reasoning: List[str]

class ValidationMetrics(BaseModel):
    key_behavioral_indicators: List[str]
    critical_evidence: List[str]
    profile_accuracy_factors: List[str]
