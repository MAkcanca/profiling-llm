/**
 * Forensic-LLM2 Static Gold Standards Loader
 *
 * This module loads gold standard profiles from static constants
 * to be compatible with GitHub Pages which doesn't support directory browsing.
 */

// Cache for gold standard profiles
let goldStandardsCache = null;

/**
 * Loads all available gold standard profiles from static constants
 * @returns {Promise<Array>} Promise resolving to gold standard profiles data
 */
async function loadGoldStandards() {
  // Return cached data if available
  if (goldStandardsCache) {
    return goldStandardsCache;
  }

  const goldStandards = [];
  
  try {
    // Get gold standards from constants
    const goldStandardsList = window.PROFILES_CONSTANTS.goldStandards;
    
    // Process each gold standard
    for (const goldStandardData of goldStandardsList) {
      try {
        // Fetch the actual JSON content for each gold standard
        const response = await fetch(goldStandardData.filePath);
        if (!response.ok) continue;
        
        const data = await response.json();
        
        // Create profile object with data from constants and extracted frameworks
        const profile = {
          id: goldStandardData.id,
          name: goldStandardData.name,
          subtitle: goldStandardData.subtitle,
          description: goldStandardData.description,
          image: goldStandardData.image,
          filePath: goldStandardData.filePath,
          frameworks: extractFrameworks(data),
          tags: [] // No tags as requested - framework classifications are displayed separately
        };
        
        goldStandards.push(profile);
      } catch (error) {
        console.error(`Error processing gold standard file ${goldStandardData.id}:`, error);
      }
    }
    
    // Sort gold standards by name
    goldStandards.sort((a, b) => a.name.localeCompare(b.name));
    
    // Store in cache
    goldStandardsCache = goldStandards;
    return goldStandards;
  } catch (error) {
    console.error('Error loading gold standards:', error);
    return [];
  }
}

/**
 * Extract framework classifications from profile data
 * @param {Object} data - The profile data
 * @returns {Object} Framework classifications
 */
function extractFrameworks(data) {
  const frameworks = {
    nas: "NOT_APPLICABLE",
    shpa: "NOT_APPLICABLE",
    bcs: "NOT_APPLICABLE",
    spatial: "NOT_APPLICABLE"
  };
  
  if (!data || !data.framework_classifications) {
    return frameworks;
  }
  
  const classifications = data.framework_classifications;
  
  // Extract NAS (Narrative Action System)
  if (classifications.narrative_action_system && classifications.narrative_action_system.primary_classification) {
    frameworks.nas = classifications.narrative_action_system.primary_classification;
  }
  
  // Extract SHPA (Sexual Behavioral Analysis)
  if (classifications.sexual_behavioral_analysis && classifications.sexual_behavioral_analysis.primary_classification) {
    frameworks.shpa = classifications.sexual_behavioral_analysis.primary_classification;
  }
  
  // Extract BCS (can be either behavioral_change_staging or sexual_homicide_pathways_analysis)
  if (classifications.sexual_homicide_pathways_analysis && classifications.sexual_homicide_pathways_analysis.primary_classification) {
    frameworks.bcs = classifications.sexual_homicide_pathways_analysis.primary_classification;
  } else if (classifications.behavioral_change_staging && classifications.behavioral_change_staging.primary_classification) {
    frameworks.bcs = classifications.behavioral_change_staging.primary_classification;
  }
  
  // Extract Spatial
  if (classifications.spatial_behavioral_analysis && classifications.spatial_behavioral_analysis.primary_classification) {
    frameworks.spatial = classifications.spatial_behavioral_analysis.primary_classification;
  }
  
  return frameworks;
}

/**
 * Format a case ID into a display name
 * @param {string} caseId - The case ID (e.g., "ted-bundy-lake")
 * @returns {string} Formatted display name (e.g., "Ted Bundy")
 */
function formatCaseName(caseId) {
  // Special case formatting
  const specialCases = {
    'ted-bundy-lake': 'Ted Bundy',
    'btk-otero': 'BTK Killer',
    'ed-kemper': 'Edmund Kemper',
    'mad-bomber': 'Mad Bomber',
    'robert-napper-rachel-nickell': 'Robert Napper',
    'unabomber': 'Unabomber'
  };
  
  if (specialCases[caseId]) {
    return specialCases[caseId];
  }
  
  // Generic formatting: Replace hyphens with spaces and capitalize words
  return caseId
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Get subtitle for a profile
 * @param {string} caseId - The case ID
 * @param {Object} data - The profile data
 * @returns {string} Subtitle
 */
function getSubtitle(caseId, data) {
  const subtitles = {
    'ted-bundy-lake': 'Lake Sammamish Murders',
    'btk-otero': 'Dennis Rader',
    'ed-kemper': 'Co-ed Killer',
    'mad-bomber': 'George Metesky',
    'robert-napper-rachel-nickell': 'Rachel Nickell Murder',
    'unabomber': 'Theodore Kaczynski'
  };
  
  return subtitles[caseId] || '';
}

/**
 * Get description for a profile
 * @param {string} caseId - The case ID
 * @returns {string} Description
 */
function getDescription(caseId) {
  const descriptions = {
    'ted-bundy-lake': 'American serial killer who kidnapped, assaulted, and murdered numerous young women during the 1970s, and possibly earlier.',
    'btk-otero': 'Serial killer who murdered ten people between 1974 and 1991 in Wichita, Kansas. Known for his self-given name \'BTK\' (Bind, Torture, Kill).',
    'ed-kemper': 'American serial killer who murdered ten people, including his paternal grandparents and mother, between 1964 and 1973.',
    'mad-bomber': 'American domestic terrorist who planted bombs in New York City between 1940 and 1957, primarily targeting utilities and theaters.',
    'robert-napper-rachel-nickell': 'British serial rapist and murderer who killed Rachel Nickell on Wimbledon Common in London in 1992.',
    'unabomber': 'American domestic terrorist and former mathematics professor who conducted a bombing campaign targeting people involved with modern technology.'
  };
  
  return descriptions[caseId] || '';
}

/**
 * Get image name for a profile
 * @param {string} caseId - The case ID
 * @returns {string} Image filename without extension
 */
function getImageName(caseId) {
  const imageNames = {
    'ted-bundy-lake': 'ted-bundy',
    'btk-otero': 'btk',
    'ed-kemper': 'ed-kemper',
    'mad-bomber': 'george-metesky',
    'robert-napper-rachel-nickell': 'robert-napper',
    'unabomber': 'theodore-kaczynski'
  };
  
  return imageNames[caseId] || caseId;
}

// Export module functions
window.GoldStandardsLoader = {
  loadGoldStandards
};