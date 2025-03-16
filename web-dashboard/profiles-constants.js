/**
 * Forensic-LLM2 Static Profile Paths
 * 
 * This file contains hardcoded paths to all profiles and gold standards
 * to make the dashboard compatible with GitHub Pages which doesn't support
 * directory browsing.
 */

const ALL_MODELS = ["Claude-3.7-Sonnet", "DeepSeek-R1", "Gemini-2.0-Flash", "Gemma-3", "GPT-4.5-Preview", "GPT-4o", "GPT-4o-mini", "Llama-3.3-70B-Instruct", "o3-mini", "o3-mini-high", "Claude-3.7-Sonnet-Thinking", "Gemini-2.0-Flash-Thinking-Exp0121"];
// Define the structure with all available profiles
window.PROFILES_CONSTANTS = {
  // Available runs with their display names
  runs: [
    {
      id: "run_20250314_201115",
      displayName: "Run 2025-03-14 20:11:15"
    },
    {
      id: "run_20250314_192949",
      displayName: "Run 2025-03-14 19:29:49"
    },
    {
      id: "run_20250314_182604",
      displayName: "Run 2025-03-14 18:26:04"
    },
    {
      id: "run_20250314_154409",
      displayName: "Run 2025-03-14 15:44:09"
    }
  ],
  
  // Test cases with their display names and available models for each run
  testCases: {
    "btk-otero": {
      displayName: "BTK Killer (Otero Family)",
      goldStandard: "gold-standards/btk-otero-profile.json",
      models: {
        "run_20250314_201115": ALL_MODELS,
        "run_20250314_192949": ALL_MODELS,
        "run_20250314_182604": ALL_MODELS,
        "run_20250314_154409": ALL_MODELS,
      }
    },
    "ed-kemper": {
      displayName: "Edmund Kemper",
      goldStandard: "gold-standards/ed-kemper-profile.json",
      models: {
        "run_20250314_201115": ALL_MODELS,
        "run_20250314_192949": ALL_MODELS,
        "run_20250314_182604": ALL_MODELS,
        "run_20250314_154409": ALL_MODELS,
      }
    },
    "mad-bomber": {
      displayName: "Mad Bomber (George Metesky)",
      goldStandard: "gold-standards/mad-bomber-profile.json",
      models: {
        "run_20250314_201115": ALL_MODELS,
        "run_20250314_192949": ALL_MODELS,
        "run_20250314_182604": ALL_MODELS,
        "run_20250314_154409": ALL_MODELS,
      }
    },
    "robert-napper-rachel-nickell": {
      displayName: "Robert Napper (Rachel Nickell)",
      goldStandard: "gold-standards/robert-napper-rachel-nickell-profile.json",
      models: {
        "run_20250314_201115": ALL_MODELS,
        "run_20250314_192949": ALL_MODELS,
        "run_20250314_182604": ALL_MODELS,
        "run_20250314_154409": ALL_MODELS,
      }
    },
    "ted-bundy-lake": {
      displayName: "Ted Bundy (Lake Sammamish)",
      goldStandard: "gold-standards/ted-bundy-lake-profile.json",
      models: {
        "run_20250314_201115": ALL_MODELS,
        "run_20250314_192949": ALL_MODELS,
        "run_20250314_182604": ALL_MODELS,
        "run_20250314_154409": ALL_MODELS,
      }
    },
    "unabomber": {
      displayName: "Unabomber (Ted Kaczynski)",
      goldStandard: "gold-standards/unabomber-profile.json",
      models: {
        "run_20250314_201115": ALL_MODELS,
        "run_20250314_192949": ALL_MODELS,
        "run_20250314_182604": ALL_MODELS,
        "run_20250314_154409": ALL_MODELS,
      }
    }
  },
  
  // Helper function to construct model file paths
  getModelFilePath: function(run, testCase, model) {
    return `generated_profiles/${run}/${testCase}/${model}_result.json`;
  }
};