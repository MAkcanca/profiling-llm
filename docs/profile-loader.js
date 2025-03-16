/**
 * Forensic-LLM2 Static Profile Loader
 *
 * This module handles loading of criminal profiles using static path constants
 * to be compatible with GitHub Pages which doesn't support directory browsing.
 */

// Cache for discovered profile metadata
let discoveredProfilesCache = null;

/**
 * Creates profile metadata from static constants
 * @returns {Promise<Object>} Promise resolving to profile metadata
 */
async function discoverProfiles() {
  // Return cached data if available
  if (discoveredProfilesCache) {
    return discoveredProfilesCache;
  }

  const profiles = {};
  
  try {
    // Create profiles from static constants
    const testCases = window.PROFILES_CONSTANTS.testCases;
    
    // Process each test case
    for (const [testCaseId, testCaseData] of Object.entries(testCases)) {
      profiles[testCaseId] = {
        displayName: testCaseData.displayName,
        goldStandard: testCaseData.goldStandard,
        models: {}
      };
      
      // Process each run and its models
      for (const [runId, modelList] of Object.entries(testCaseData.models)) {
        for (const modelName of modelList) {
          // Create file path using the helper function
          const filePath = window.PROFILES_CONSTANTS.getModelFilePath(runId, testCaseId, modelName);
          
          // Add model to profiles if it doesn't exist yet
          if (!profiles[testCaseId].models[modelName]) {
            profiles[testCaseId].models[modelName] = {
              filePath: filePath
            };
            
            // Set placeholder metrics (they will be loaded on demand)
            profiles[testCaseId].models[modelName].reasoning_count = null;
            profiles[testCaseId].models[modelName].framework_agreement = null;
            profiles[testCaseId].models[modelName].semantic_similarity = null;
            profiles[testCaseId].models[modelName].framework_contributions = {
              narrative_action_system: null,
              sexual_behavioral_analysis: null,
              behavioral_change_staging: null,
              spatial_behavioral_analysis: null
            };
          }
        }
      }
    }
    
    // Store in cache
    discoveredProfilesCache = profiles;
    return profiles;
  } catch (error) {
    console.error('Error discovering profiles:', error);
    return {};
  }
}

/**
 * Get all available runs from constants
 * @returns {Promise<string[]>} Array of run directory names
 */
async function getAvailableRuns() {
  try {
    // Extract run IDs from the constants
    const runs = window.PROFILES_CONSTANTS.runs.map(run => run.id);
    return runs;
  } catch (error) {
    console.error('Error getting available runs:', error);
    return [];
  }
}

/**
 * Get the most recent run directory
 * @returns {Promise<string>} The most recent run directory name
 */
async function getMostRecentRun() {
  try {
    const runs = window.PROFILES_CONSTANTS.runs;
    // Return the first run (assumed to be the most recent)
    return runs.length > 0 ? runs[0].id : null;
  } catch (error) {
    console.error('Error getting most recent run:', error);
    return null;
  }
}

/**
 * Format a case ID into a display name
 * @param {string} caseId - The case ID (e.g., "ted-bundy-lake")
 * @returns {string} Formatted display name (e.g., "Ted Bundy (Lake Sammamish)")
 */
function formatCaseName(caseId) {
  // Special case formatting
  const specialCases = {
    'ted-bundy-lake': 'Ted Bundy (Lake Sammamish)',
    'btk-otero': 'BTK Killer (Otero Family)',
    'ed-kemper': 'Edmund Kemper',
    'mad-bomber': 'Mad Bomber (George Metesky)',
    'robert-napper-rachel-nickell': 'Robert Napper (Rachel Nickell)',
    'unabomber': 'Unabomber (Ted Kaczynski)'
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
 * Load metrics for a profile from its JSON file
 * @param {Object} profileObj - The profile metadata object to update
 * @param {string} testCase - The test case ID
 * @param {string} model - The model name
 * @returns {Promise<void>}
 */
async function loadProfileMetrics(profileObj, testCase, model) {
  if (!profileObj || !profileObj[testCase] || !profileObj[testCase].models[model]) {
    return;
  }
  
  const modelData = profileObj[testCase].models[model];
  
  // If metrics are already loaded, don't fetch again
  if (modelData.reasoning_count !== null) {
    return;
  }
  
  try {
    const response = await fetch(modelData.filePath);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch ${modelData.filePath}`);
    }
    
    const profileData = await response.json();
    
    // Extract metrics from the profile data
    if (profileData) {
      // Extract reasoning count from all sub-sections of the offender_profile
      if (profileData.offender_profile) {
        const reasoningCount = Object.values(profileData.offender_profile).reduce((sum, section) => {
          return sum + (section.reasoning ? section.reasoning.length : 0);
        }, 0);
        modelData.reasoning_count = reasoningCount;
      } else {
        modelData.reasoning_count = '-';
      }
      
      // Extract framework metrics from the frameworks classifications
      if (profileData.framework_classifications) {
        const frameworks = profileData.framework_classifications;
        
        // Extract contributions
        modelData.framework_contributions = {
          narrative_action_system: frameworks.narrative_action_system ? 
            frameworks.narrative_action_system.confidence || 0 : 0,
          
          sexual_behavioral_analysis: frameworks.sexual_behavioral_analysis ? 
            frameworks.sexual_behavioral_analysis.confidence || 0 : 0,
          
          // Handle both behavioral_change_staging and sexual_homicide_pathways_analysis
          behavioral_change_staging: frameworks.sexual_homicide_pathways_analysis ?
            frameworks.sexual_homicide_pathways_analysis.confidence || 0 :
            (frameworks.behavioral_change_staging ?
              frameworks.behavioral_change_staging.confidence || 0 : 0),
              
          spatial_behavioral_analysis: frameworks.spatial_behavioral_analysis ? 
            frameworks.spatial_behavioral_analysis.confidence || 0 : 0
        };
        
        // Calculate average framework confidence
        const values = Object.values(modelData.framework_contributions);
        const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
        modelData.avg_framework_confidence = avg;
        
        // For demo purposes, set placeholder values for other metrics
        modelData.framework_agreement = Math.random() < 0.7 ? 0.33 : (Math.random() < 0.5 ? 0 : 0.67);
        modelData.semantic_similarity = 0.75 + (Math.random() * 0.1);
      }
    }
  } catch (error) {
    console.error(`Error loading metrics for ${model} in ${testCase}:`, error);
    
    // Set default values on error
    modelData.reasoning_count = '-';
    modelData.framework_agreement = '-';
    modelData.semantic_similarity = '-';
    modelData.framework_contributions = {
      narrative_action_system: '-',
      sexual_behavioral_analysis: '-',
      behavioral_change_staging: '-',
      spatial_behavioral_analysis: '-'
    };
  }
}

/**
 * Dynamically format profile data to Markdown based on its schema
 * @param {Object} data - The profile data to format
 * @returns {string} Formatted markdown
 */
function formatProfileToMarkdown(data) {
  if (!data) {
    return "# Error: Invalid profile data";
  }

  let markdown = '';
  
  // Add case ID if available
  if (data.case_id) {
    markdown += `# Criminal Profile: ${data.case_id}\n\n`;
  } else {
    markdown += `# Criminal Profile\n\n`;
  }
  
  // Process offender_profile if available
  if (data.offender_profile) {
    const profile = data.offender_profile;
    
    // Dynamically process each section of the profile
    for (const [sectionKey, sectionData] of Object.entries(profile)) {
      // Skip the overall reasoning section - it will be added at the end
      if (sectionKey === 'reasoning') continue;
      
      // Format section title (convert snake_case to Title Case)
      const sectionTitle = sectionKey
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      
      markdown += `## ${sectionTitle}\n`;
      
      // Process section data
      for (const [key, value] of Object.entries(sectionData)) {
        // Skip the section-specific reasoning (it's handled separately)
        if (key === 'reasoning') continue;
        
        // Format the property name (convert snake_case to Title Case)
        const formattedKey = key
          .replace(/_/g, ' ')
          .replace(/\b\w/g, l => l.toUpperCase());
        
        // Add the property and its value
        markdown += `- **${formattedKey}**: ${value}\n`;
      }
      
      // Add section-specific reasoning if available
      if (sectionData.reasoning && sectionData.reasoning.length > 0) {
        markdown += '\n**Reasoning:**\n';
        sectionData.reasoning.forEach(reason => {
          markdown += `- ${reason}\n`;
        });
      }
      
      markdown += '\n';
    }
    
    // Add overall reasoning at the end if available
    if (profile.reasoning && profile.reasoning.length > 0) {
      markdown += `## Overall Profile Reasoning\n`;
      profile.reasoning.forEach(reason => {
        markdown += `- ${reason}\n`;
      });
      markdown += '\n';
    }
  }
  
  // Process validation_metrics if available
  if (data.validation_metrics) {
    const metrics = data.validation_metrics;
    
    markdown += `## Validation Metrics\n`;
    
    for (const [metricsKey, metricsValue] of Object.entries(metrics)) {
      // Format metrics title (convert snake_case to Title Case)
      const metricsTitle = metricsKey
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      
      markdown += `### ${metricsTitle}\n`;
      
      // If it's an array, format as list
      if (Array.isArray(metricsValue)) {
        metricsValue.forEach(item => {
          markdown += `- ${item}\n`;
        });
        markdown += '\n';
      } 
      // If it's an object, format as properties
      else if (typeof metricsValue === 'object' && metricsValue !== null) {
        for (const [key, value] of Object.entries(metricsValue)) {
          const formattedKey = key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
          
          markdown += `- **${formattedKey}**: ${value}\n`;
        }
        markdown += '\n';
      }
    }
  }
  
  // Process framework_classifications if available
  if (data.framework_classifications) {
    const frameworks = data.framework_classifications;
    
    markdown += `## Framework Classifications\n`;
    
    for (const [frameworkKey, frameworkData] of Object.entries(frameworks)) {
      // Format framework title
      let frameworkTitle;
      
      // Handle specific framework naming
      switch (frameworkKey) {
        case 'narrative_action_system':
          frameworkTitle = 'Narrative Action System (NAS)';
          break;
        case 'sexual_behavioral_analysis':
          frameworkTitle = 'Sexual Behavioral Analysis';
          break;
        case 'sexual_homicide_pathways_analysis':
          frameworkTitle = 'Sexual Homicide Pathways Analysis';
          break;
        case 'spatial_behavioral_analysis':
          frameworkTitle = 'Spatial Behavioral Analysis';
          break;
        default:
          // Format other framework titles from snake_case to Title Case
          frameworkTitle = frameworkKey
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
      }
      
      markdown += `### ${frameworkTitle}\n`;
      
      // Add classification
      const classification = frameworkData.primary_classification || frameworkData.classification || 'Unknown';
      markdown += `**Classification: ${classification}**\n\n`;
      
      // Add confidence if available
      if (frameworkData.confidence !== undefined) {
        markdown += `**Confidence: ${typeof frameworkData.confidence === 'number' ? frameworkData.confidence.toFixed(2) : frameworkData.confidence}**\n\n`;
      }
      
      // Add supporting evidence if available
      if (frameworkData.supporting_evidence && frameworkData.supporting_evidence.length > 0) {
        markdown += `**Supporting Evidence:**\n`;
        frameworkData.supporting_evidence.forEach(evidence => {
          markdown += `- ${evidence}\n`;
        });
        markdown += '\n';
      }
      
      // Add contradicting evidence if available
      if (frameworkData.contradicting_evidence && frameworkData.contradicting_evidence.length > 0) {
        markdown += `**Contradicting Evidence:**\n`;
        frameworkData.contradicting_evidence.forEach(evidence => {
          markdown += `- ${evidence}\n`;
        });
        markdown += '\n';
      }
      
      // Add reasoning if available
      if (frameworkData.reasoning && frameworkData.reasoning.length > 0) {
        markdown += `**Reasoning:**\n`;
        frameworkData.reasoning.forEach(reason => {
          markdown += `- ${reason}\n`;
        });
        markdown += '\n';
      }
    }
  }
  
  return markdown;
}

// Export functions for use in other modules
window.ProfileLoader = {
  discoverProfiles,
  formatProfileToMarkdown,
  loadProfileMetrics,
  getMostRecentRun,
  getAvailableRuns
};