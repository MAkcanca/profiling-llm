enum NarrativeActionSystemType {
  VICTIM
  PROFESSIONAL
  TRAGIC_HERO
  REVENGEFUL
}

enum SexualBehavioralAnalysisType {
  POWER_ASSERTIVE
  POWER_REASSURANCE
  ANGER_RETALIATORY
  ANGER_EXCITATION
}

enum SexualHomicidePathwaysType {
  SADISTIC
  ANGRY
  OPPORTUNISTIC
}

enum SpatialBehaviorType {
  MARAUDER
  COMMUTER
}

enum Gender {
  MALE
  FEMALE
}

// We are aliasing field names to symbols like "k1" ... "kN".
// This makes the LLM pay attention to your descriptions instead of just the enum or property name, since those can introduce biases. See paper: https://arxiv.org/abs/2305.08298 for more details.

class Demographics {
  age_range string @description("Estimated age range of the subject") @alias("k1")
  gender Gender @description("Gender of the subject") @alias("k2")
  employment string @description("Type of employment or work") @alias("k3")
  relationship string @description("Relationship or marital status") @alias("k4")
  living_situation string @description("Living arrangements of the subject") @alias("k5")
  education string @description("Educational background and intelligence level") @alias("k6")
  reasoning string[]
}

class PsychologicalCharacteristics {
  personality_type string @description("General personality classification") @alias("k7")
  control_needs string @description("Need for power and control assessment") @alias("k8")
  social_competence string @description("Social skills and ability to maintain relationships") @alias("k9")
  stress_factors string @description("Likely stressors affecting the subject") @alias("k10")
  fantasy_elements string @description("Nature and intensity of fantasies") @alias("k11")
  anger_management string @description("How anger is expressed and controlled") @alias("k12")
  reasoning string[]
}

class BehavioralCharacteristics {
  crime_scene_behavior string @description("Organization and methodology at crime scene") @alias("k13")
  violence_type string @description("Nature and pattern of violence") @alias("k14")
  victim_interaction string @description("How the subject interacts with victims") @alias("k15")
  post_offense_behavior string @description("Behavior after committing offense") @alias("k16")
  risk_taking string @description("Level and nature of risks taken") @alias("k17")
  trophy_taking string @description("Whether and what kind of trophies are taken") @alias("k18")
  reasoning string[]
}

class GeographicBehavior {
  crime_scene_location string @description("Location of the crime scene") @alias("k19")
  crime_scene_layout string @description("Layout of the crime scene") @alias("k20")
  crime_scene_characteristics string @description("Characteristics of the crime scene") @alias("k21")
  reasoning string[]
}

class SkillsAndKnowledge {
  technical_skills string @description("Level of technical proficiency") @alias("k22")
  planning_ability string @description("Capacity for planning and organization") @alias("k23")
  weapon_proficiency string @description("Comfort and skill with weapons") @alias("k24")
  knot_tying string @description("Skill level with restraints or knots if applicable") @alias("k25")
  crime_scene_awareness string @description("Awareness of forensic evidence") @alias("k26")
  reasoning string[]
}

class InvestigativeImplications {
  prior_offenses string @description("Likely history of previous crimes") @alias("k27")
  escalation_risk string @description("Risk of escalation in future crimes") @alias("k28")
  documentation string @description("Whether subject documents crimes") @alias("k29")
  victim_selection string @description("Pattern in victim selection, victimology evidence") @alias("k30")
  cooling_off_period string @description("Time between offenses") @alias("k31")
  reasoning string[]
}

class KeyIdentifiers {
  occupation_type string @description("Likely occupation or job position") @alias("k32")
  appearance string @description("Physical appearance and presentation") @alias("k33")
  social_role string @description("Role in community or social groups") @alias("k34")
  hobbies string @description("Likely interests or hobbies") @alias("k35")
  vehicle string @description("Type or condition of vehicle") @alias("k36")
  reasoning string[]
}

class ValidationMetrics {
  key_behavioral_indicators string[] @description("Key behaviors indicating profile match") @alias("k37")
  critical_evidence string[] @description("Critical evidence supporting the profile") @alias("k38")
  profile_accuracy_factors string[] @description("Factors determining profile accuracy") @alias("k39")
}

class FrameworkClassifications {
  narrative_action_system NarrativeActionSystemClassification @description("NAS classification") @alias("k40")
  sexual_behavioral_analysis SexualBehavioralAnalysisClassification @description("SHPA classification (if applicable)") @alias("k41")
  sexual_homicide_pathways_analysis SexualHomicidePathwaysClassification @description("SHPA classification (if applicable)") @alias("p41")
  spatial_behavioral_analysis SpatialBehaviorClassification @description("Spatial behavior classification") @alias("k43")
}

class NarrativeActionSystemClassification {
  primary_classification NarrativeActionSystemType @description("Primary NAS classification") @alias("k44")
  confidence float @description("Confidence score (0.0 to 1.0)") @alias("k45")
  supporting_evidence string[] @description("Evidence supporting this classification") @alias("k46")
  contradicting_evidence string[] @description("Evidence contradicting this classification") @alias("k47")
  reasoning string[] @description("Reasoning for this classification") @alias("k48")
}

class SexualBehavioralAnalysisClassification {
  primary_classification SexualBehavioralAnalysisType? @description("Primary SHPA classification") @alias("k49")
  confidence float? @description("Confidence score (0.0 to 1.0)") @alias("k50")
  supporting_evidence string[]? @description("Evidence supporting this classification") @alias("k51")
  contradicting_evidence string[]? @description("Evidence contradicting this classification") @alias("k52")
  reasoning string[]? @description("Reasoning for this classification") @alias("k53")
}

class SexualHomicidePathwaysClassification {
  primary_classification SexualHomicidePathwaysType? @description("Primary Sexual Pathways classification") @alias("p54")
  confidence float? @description("Confidence score (0.0 to 1.0)") @alias("p55")
  supporting_evidence string[]? @description("Evidence supporting this classification") @alias("p56")
  contradicting_evidence string[]? @description("Evidence contradicting this classification") @alias("p57")
  reasoning string[]? @description("Reasoning for this classification") @alias("p58")
}

class SpatialBehaviorClassification {
  primary_classification SpatialBehaviorType @description("Primary spatial behavior classification") @alias("k59")
  confidence float @description("Confidence score (0.0 to 1.0)") @alias("k60")
  supporting_evidence string[] @description("Evidence supporting this classification") @alias("k61")
  contradicting_evidence string[] @description("Evidence contradicting this classification") @alias("k62")
  reasoning string[] @description("Reasoning for this classification") @alias("k63")
}

class OffenderProfile {
  demographics Demographics @description("Demographic information about the subject") @alias("k64")
  psychological_characteristics PsychologicalCharacteristics @description("Psychological traits and patterns") @alias("k65")
  behavioral_characteristics BehavioralCharacteristics @description("Behavioral patterns and crime scene conduct") @alias("k66")
  geographic_behavior GeographicBehavior @description("Geographic patterns and spatial behavior") @alias("k67")
  skills_and_knowledge SkillsAndKnowledge @description("Skills, abilities, and knowledge of the subject") @alias("k68")
  investigative_implications InvestigativeImplications @description("Implications for investigation strategies") @alias("k69")
  key_identifiers KeyIdentifiers @description("Key identifying characteristics") @alias("k70")
  reasoning string[] @description("Reasoning for this profile") @alias("k71")
}

class ProfileResult {
  case_id string @description("case_id from the case data") @alias("k72")
  offender_profile OffenderProfile @description("Comprehensive offender profile analysis") @alias("k73")
  validation_metrics ValidationMetrics @description("Metrics for validating the profile") @alias("k74")
  framework_classifications FrameworkClassifications @description("Classifications according to established frameworks") @alias("k75")
}

template_string SystemMessage() #"
    You are an expert criminal profiler specializing in psychological and behavioral analysis. Use the evidence assessment to create a comprehensive offender profile following the structured format. Be evidence-based, methodical, and specific in your analysis, drawing connections between evidence and profile elements.Reason step by step, in the final answer, provide the reasoning for each step.
    # CRIMINAL PROFILING PROCESS

    To create a comprehensive offender profile for this case, follow these steps:
    1. Review the evidence which contains:
    - Crime scene evidence
    - Victimology evidence
    - Behavioral evidence
    - Geographic evidence
    - Timeline evidence
    - Communication evidence
    - Framework indicators
    - Critical missing evidence
    2. Analyze this evidence using the profiling frameworks
    3. Create a comprehensive offender profile that includes demographic, psychological, behavioral, geographic, skills, investigative implications, and identifying characteristics
    4. Include framework classifications with confidence ratings
    5. Validate your profile with key indicators and critical evidence

    ## Profiling Frameworks

    Apply these established profiling methodologies in your analysis:

    1. Narrative Action System (NAS) Archetypes:

    REVENGEFUL MISSION (framework_id: nas_revengeful):
    Description: Strong protagonist seeking specific impact on significant others
    Behavioral Indicators:
    - Targeted victim selection with personal significance
    - Controlled, methodical violence without signs of chaos
    - Evidence of preparation and determination to complete action
    - Crime scene shows signs of a justified "mission" or purpose
    - Calm execution despite negative motivation

    TRAGIC HERO (framework_id: nas_tragic):
    Description: Powerless protagonist acting out inevitable fate
    Behavioral Indicators:
    - Chaotic crime scene suggesting lack of control
    - Evidence of perceived justification or "no choice" scenario
    - Minimal concern for victim impact or consequences
    - Signs of being "swept along" by circumstances
    - Actions framed as unavoidable response to situation

    PROFESSIONAL (framework_id: nas_professional):
    Description: Competent protagonist demonstrating mastery and control
    Behavioral Indicators:
    - Efficient crime scene with minimal unnecessary actions
    - Technical skill evident in execution of crime
    - Target selection based on practical considerations, not personal
    - Evidence of planning, risk assessment and exit strategy
    - Calm, unemotional approach focused on achievement

    VICTIM (framework_id: nas_victim):
    Description: Confused protagonist acting from perceived powerlessness
    Behavioral Indicators:
    - Disorganized actions showing emotional arousal
    - Evidence of perceived provocation or persecution
    - Signs of feeling "forced" into criminal action
    - Crime scene reflects confusion and lack of planning
    - Clear significance of others in motivating the behavior
    - Pattern of blaming external forces or people

    2.1 Sexual Homicide Patterns Analysis (SHPA):

    ANGER_EXCITATION (framework_id: shpa_anger_excitation):
    Description: Planned sexual assault and murder designed to inflict pain and terror for perpetrator's gratification.

    Indicators:
    - Extended torture or bondage
    - Both rape and murder are planned
    - Prolonged torture of victim for sexual gratification
    - Satisfaction from process of killing, not death itself
    - Ritualistic carnage driven by specialized fantasies
    - Domination and control as primary motivation

    Behavioral Indicators:
    - Sophisticated restraints
    - Extensive planning
    - Extended time with victim
    - Sexual positioning
    - Evidence of sexual fantasy enactment
  
    Crime Scene Characteristics:
    - Evidence of prolonged, bizarre ritualistic assault
    - Uses con or ruse to isolate victim
    - Bondage and domination elements
    - Antemortem cuttings, bruises, strangulation
    - Postmortem sexual activity and experimentation
    - May take body parts as souvenirs
    - Carefully packed murder kit with specialized tools
    - Body possibly moved to second location
    - Highly organized crime scene

    Offender Characteristics:
    - Variable age range, typically before 35
    - Well-appearing, bright, socially skilled
    - Can appear conventional and law-abiding
    - May maintain normal marriage/relationships
    - Compulsive and structured personality
    - Possible mechanical interests
    - Private area for paraphernalia and souvenirs
    - Preference for bondage and sadism pornography
    - Extensive fantasy life
    - May inject himself into investigation


    ANGER_RETALIATORY (framework_id: shpa_anger_retaliatory):
    Description: Planned sexual assault and murder as symbolic revenge and anger
    Indicators:
    - Blitz attack
    - Minimal planning
    - Both rape and murder are planned
    - Anger-driven response to humiliation or rejection by women
    - Expresses revenge through stylized violent attack
    - Seeks catharsis through explosive violence
    - Sexual acts as punishment

    Behavioral Indicators:
    - Facial battery
    - Multiple weapons
    - Disorganized scene
    - Personal items destroyed
    - Evidence of rage
  
    Crime Scene Characteristics:
        - Overkill with extreme violence
        - Brutal beating of face and mouth
        - May use weapons of opportunity
        - Incomplete rape due to anger/erectile dysfunction
        - Body placed in submissive position after death
        - Disorganized crime scene
        - Improvised murder weapon found near body
        - Often takes small trinket or souvenir
        - Blames victim for the murder

    Offender Characteristics:
        - Mid-to-late 20s
        - Explosive, impulsive personality
        - Quick-tempered and self-centered
        - Superficial social relationships
        - Conflicted relationship with women
        - History of spousal abuse
        - Sexually frustrated, possibly impotent
        - Links eroticized anger with sexual competence
        - Criminal history of assault and domestic violence
        - School dropout with authority problems


    POWER_REASSURANCE (framework_id: shpa_power_reassurance):
    Description: Planned rape followed by unplanned overkill when fantasy of seduction fails.
    Indicators:
    - Rape is planned; murder is not planned
    - Motivated by idealized seduction and conquest fantasy
    - Seeks verbal reassurance of sexual adequacy
    - Panics into murder when victim doesn't comply with fantasy
    - May explore sexual curiosities through postmortem activity

    Behavioral Indicators:
    - Evidence of fantasy rehearsal
    - Communication attempts with victim
    - Taking small mementos
    - Minimal necessary force until fantasy is disrupted
    - Nighttime activity preferences
    - Return to comfort zone after crime
  
    Crime Scene Characteristics:
    - Victim selected and watched beforehand
    - Acts out pre-programmed fantasy
    - Polite verbal dialogue attempted ("gentleman rapist")
    - Murder through pummeling and manual strangulation
    - Sometimes mutilation coupled with ritualism
    - Often no evidence of sperm at scene
    - May collect souvenirs and newspaper clippings
    - Acts during nighttime hours
    - Disorganized crime scene

    Offender Characteristics:
    - Typically mid-20s
    - Socially isolated with no friends
    - Viewed as a loner or "weirdo"
    - Unmarried without normal sexual history
    - Underachiever with possible learning disability
    - Extensive rape fantasies and window peeping
    - Uses pornography and daydreaming
    - Immature, life spectator not participant
    - May live at home with minimal income
    - History of fetish activities


    POWER_ASSERTIVE (framework_id: shpa_power_assertive):
    Description: Sexual assault where rape is planned but murder is an unplanned response to maintain control.

    Indicators:
    - Rape is planned; murder is not planned
    - Demonstrates mastery through assertive image and dominating violence
    - Increasing aggression with victim ensures control
    - Focused on expressing virility, masculinity, and dominance
    - Multiple antemortem rapes often occur

    Behavioral Indicators:
    - Binding/restraints
    - Controlled scene
    - Evidence of domination
    - Extended time at scene
    - Specific victim selection
  
    Crime Scene Characteristics:
    - Victim selected by opportunity and surprise
    - Clothing torn off victim
    - Evidence of beating and pummeling
    - Organized crime scene
    - No postmortem mutilation
    - Weapon brought to and taken from scene
    - Body left undisturbed if assault occurred on victim's territory

    Offender Characteristics:
    - Usually early 20s
    - Macho image projection (often muscular, tattoos)
    - Arrogant, condescending attitude
    - History of property crimes (burglary, theft, robbery)
    - School dropout
    - Possible military service with poor record
    - Limited pornography interest
    - Strong antihomosexual attitudes
    - May brag about crime to establish status

    2.2 Sexual Homicide Pathways Analysis (Pathways):
    PATHWAY 1: SADISTIC PATHWAY (framework_id: shpaf_sadistic)
    Description: Sexual homicide committed through planned, calculated means to fulfill sadistic fantasies and achieve sexual gratification through control, suffering, and death.
    Primary Motivation: Sexual gratification through power, domination, and victim suffering
    Pre-Crime Characteristics:

    - Extensive planning and preparation
    - Victim selection based on specific criteria aligning with fantasy
    - Offense-specific tools/weapons prepared in advance
    - Calm, methodical emotional state before the offense
    - Detailed sadistic fantasy development over time
    - Evidence of sexual preoccupation and deviant sexual interests
    - Often consumes sadistic pornography

    Crime Scene Indicators:

    - Evidence of extended victim captivity
    - Sophisticated restraints and/or bondage
    - Torture implements and/or torture injuries
    - Sexual positioning of the body
    - Ritualistic elements (patterns, symbols, sequences of actions)
    - Post-mortem sexual activity
    - Meticulous crime scene (staged or cleaned)
    - Body disposal planned and executed carefully
    - Trophy-taking or souvenirs from victims
    - Erotic/sexual mutilation

    Offender Characteristics:

    - Often appears socially functional
    - Above-average intelligence
    - History of paraphilias and sexual deviance
    - Emotional detachment and callousness
    - Organized lifestyle
    - Limited criminal history prior to murder
    - Developmental history often includes humiliation and attachment issues
    - Psychopathic traits, particularly emotional detachment
    - Sexual sadism diagnosis common

    PATHWAY 2: ANGRY PATHWAY (framework_id: shpaf_angry)
    Description: Sexual homicide resulting from intense anger, rage, or vengeance, where sexual elements serve as vehicles for expression of aggression rather than primary gratification.
    Primary Motivation: Expression of rage, revenge, or perceived grievances
    Pre-Crime Characteristics:

    - Limited planning beyond target selection
    - Triggering event often present (rejection, conflict, perceived slight)
    - Intense negative emotional state prior to offense (anger, humiliation)
    - Alcohol/substance use common before attack
    - History of anger management problems
    - Previous conflict with victim or victim type

    Crime Scene Indicators:

    - Excessive violence beyond what needed to cause death
    - Overkill - multiple wounds, continued assault after death
    - Facial battery (attacking the identity)
    - Weapon of opportunity or brought for intimidation
    - Disorganized crime scene
    - Minimal attempt to conceal evidence
    - Sexual assault may be functionally punitive rather than gratifying
    - Body often left at assault location
    - Minimal post-mortem interaction
    - Degrading/humiliating assault elements

    Offender Characteristics:

    - History of violent behavior
    - Emotional dysregulation issues
    - Prior intimate partner violence common
    - Hostility toward women or specific groups
    - Antisocial personality features
    - Criminal versatility
    - Impulsivity and poor behavioral controls
    - History of abuse or rejection
    - Lower intelligence than sadistic pathway

    PATHWAY 3: SEXUALLY OPPORTUNISTIC PATHWAY (framework_id: shpaf_opportunistic)
    Description: Sexual homicide occurring as an extension of a sexual assault, where homicide is instrumental (not planned) to silence the victim, avoid detection, or in response to unexpected victim resistance.
    Primary Motivation: Sexual gratification initially, with homicide serving instrumental purposes
    Pre-Crime Characteristics:

    - Minimal planning of homicide component
    - Initial intent focused on sexual assault only
    - Moderate emotional arousal prior to offense
    - Often impaired by substance use
    - Often has history of non-homicidal sexual offenses
    - Decision to kill made during the offense

    Crime Scene Indicators:

    - Evidence of victim resistance
    - Sudden escalation of violence
    - Less ritualistic elements than sadistic pathway
    - Cause of death often manual or ligature strangulation
    - Sexual assault typically occurs pre-mortem
    - Minimal post-mortem interaction
    - Basic attempts to avoid detection (concealing body, removing evidence)
    - Weapon use primarily for control rather than torture
    - Disorganized elements mixed with some organization

    Offender Characteristics:

    - Criminal versatility
    - History of sexual offenses without homicide
    - Moderate psychopathic traits
    - Mixed personality pathology
    - Impulsivity and poor problem-solving skills
    - Often younger than other pathways
    - Less sophisticated fantasy development
    - Situational and contextual factors highly influential
  
    3. Spatial Behavioral Analysis:
    Make sure to use the following indicators and provide confidence along with the reasoning as shown.
      - Spatial Dispersion: Marauders show clustering around anchor points; commuters show clear separation between anchor and crime locations
      - Buffer Zone Analysis: Marauders typically avoid immediate home area; commuters show no pattern relative to home area
      - Transportation Evidence: Marauders often travel by foot or minimal transportation; commuters demonstrate consistent transportation methods over longer distances
      - Environmental Knowledge: Marauders exploit detailed local knowledge; commuters select environments based on specific attributes regardless of familiarity
      - Target Selection: Marauders often select opportunistic targets within zone; commuters may travel specifically for particular target types

    MARAUDER (framework_id: spatial_marauder):
    Description: Offender operates from a fixed geographic anchor point (typically residence) and commits crimes within their established comfort zone; demonstrates spatial behavior consistent with concentric zone patterns
    Indicators:
    - Crime locations form a clustered pattern around a central anchor point
    - Evidence of localized environmental knowledge and familiarity
    - Presence of a "buffer zone" immediately surrounding residence where crimes are avoided
    - Consistent return to familiar territories for criminal activity
    - Limited transportation requirements or evidence of walking to crime scenes
    - Selection of targets based on proximity rather than specific characteristics

    COMMUTER (framework_id: spatial_commuter):
    Description: Offender travels outside their home/anchor area to commit crimes; maintains deliberate geographic separation between personal and criminal domains
    Indicators:
    - Clear spatial distinction between residence/work zones and offense locations
    - Willingness to cross jurisdictional boundaries to commit crimes
    - Evidence of targeted location selection based on specific environmental variables
    - Consistent travel patterns to crime scenes (particular routes or transportation methods)
    - Absence of spatial clustering around residential areas
    - Crimes may demonstrate consistency in environmental characteristics despite geographic dispersion
    - Greater average distance between anchor points and crime scenes compared to marauder pattern
"#

// Examine {{ _.role("system") }} because it was causing issues with the models like deepseek
template_string CaseDataTemplate(case_data: string) #"
    {{ _.role("system") }}
    {{ SystemMessage() }}
    {{ _.role("user") }}
    You are an expert criminal profiler specializing in psychological and behavioral analysis. Use the evidence assessment to create a comprehensive offender profile following the structured format. Be evidence-based, methodical, and specific in your analysis, drawing connections between evidence and profile elements.Reason step by step, in the final answer, provide the reasoning for each step.
    Extract a profile from the following case data:


    CASE_DATA:
    {{ case_data }}


    {{ ctx.output_format }}
    Before you output the JSON, please explain your
    reasoning step-by-step. Here is an example on how to do this:
    'If we think step by step we can see that ...
     therefore the final output JSON is:
    {
      ... the json schema ...
    }'
"#

function ExtractProfile(case_data: string) -> ProfileResult {
  client OpenRouter
  prompt #"{{CaseDataTemplate(case_data)}}"#
}

// Test when the input is an image
test ProfileTest {
  functions [ExtractProfile]
  args {
    case_data #"
case_id: 9dd8bd29-8376-4deb-8de0-bc8be2cf04b3

INCIDENT SUMMARY
On November 12, 1989, at approximately 10:15 AM EST, a package delivered through the United States Postal Service detonated at the offices of Northwood Research Institute. The package was addressed to Dr. Robert Garrison, who opened it at his desk. The explosion resulted in significant injuries to Dr. Garrison and minor injuries to an administrative assistant in an adjacent office.
VICTIM INFORMATION

Name: Dr. Robert Garrison, Ph.D.
Age: 42
Occupation: Senior Researcher, Northwood Research Institute
Professional Background: Computer science researcher specializing in pattern recognition algorithms
Medical Status: Victim sustained injuries to hands, face, and torso. Currently in stable condition at Massachusetts General Hospital.

SCENE EXAMINATION
Location Details

Mid-size research office on third floor of five-story building
Package was opened at victim's desk in private office
Blast damage confined primarily to immediate desk area with window damage
Adjacent offices sustained minor damage from pressure wave

Physical Evidence Recovered

Package Remnants:

Brown kraft paper fragments with adhesive residue
Partial shipping label with typed address
Postal markings indicating processing through Oakland, CA distribution center
Postmarked November 8, 1989


Device Components:

Fragments of hand-crafted wooden box (approximately 8" x 6" x 2")
Battery components (consistent with 9-volt type)
Copper wire fragments with distinctive cutting pattern
Spring mechanism components
Metal pipe fragments (appears to be main explosive container)
Metal fragments (common nails and screws, likely added as shrapnel)


Trace Evidence:

Wood splinters (preliminary analysis indicates pine with hand-planed surface)
Explosive residue samples collected from multiple surfaces
Adhesive samples from package sealing
Tool mark impressions in wooden components



DEVICE CHARACTERISTICS
Based on recovered components and witness statements, the device appears to be:

Contained within a handcrafted wooden box
Triggered by opening the package/lid
Constructed primarily of common, untraceable materials
Assembled with considerable technical skill and craftsmanship
Designed to cause maximum injury to the person opening the package

CHAIN OF CUSTODY

Scene secured by Cambridge PD at 10:23 AM
FBI Evidence Response Team arrived at 11:45 AM
Scene processed until 19:30 hours
All evidence transported to FBI Laboratory, Quantico under standard protocols
Evidence log maintained by SA Thompson #9231

WITNESS STATEMENTS
Martha Collins (Administrative Assistant)

Observed delivery of package with regular mail at approximately 9:30 AM
Package appeared "professionally wrapped" with typed address label
Delivered package to Dr. Garrison's office at approximately 10:10 AM
Was in adjacent office when explosion occurred
No observations of suspicious persons in building

Dr. James Werner (Colleague)

Was meeting with Dr. Garrison 30 minutes before explosion
Did not recall seeing package on desk during meeting
Reported that Dr. Garrison had recently published controversial paper on surveillance recognition technology
Not present at time of explosion

Delivery Personnel

Regular postal carrier interviewed - followed standard delivery procedures
No recollection of specific package
Route surveillance cameras being requested for review

SIMILAR INCIDENTS
This incident bears similarities to six previous mail bombing cases reported between 1985-1989:

Northwestern University (1985) - Engineering Department
University of Michigan (1985) - Research Laboratory
California Technology Firm (1987) - Executive Office
Utah Biological Sciences Center (1987) - Researcher's Home
Massachusetts Institute of Technology (1987) - Faculty Office
Pennsylvania Robotics Laboratory (1988) - Package Intercepted

Common elements in these cases include:

Academic or technological industry targets
Package delivery method
Handcrafted wooden containers
Similar explosive mechanisms
No recoverable fingerprints or DNA
No clear personal connection between victims

PRELIMINARY FINDINGS

Device constructed with methodical attention to detail
No electrical or factory-made timing mechanisms
No fingerprints or other biological evidence recovered
No explicit messages or communications included with device
Materials appear deliberately selected to be untraceable
Postal routing suggests mailing origin in Northern California area


    "#

  }
}
