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
  
  // Gold standard files with their metadata
  goldStandards: [
    {
      id: "ted-bundy-lake",
      name: "Ted Bundy",
      subtitle: "Lake Sammamish Murders",
      description: "American serial killer who kidnapped, assaulted, and murdered numerous young women during the 1970s, and possibly earlier.",
      image: "images/criminal-profiles/ted-bundy.jpg",
      filePath: "gold-standards/ted-bundy-lake-profile.json"
    },
    {
      id: "btk-otero",
      name: "BTK Killer",
      subtitle: "Dennis Rader",
      description: "Serial killer who murdered ten people between 1974 and 1991 in Wichita, Kansas. Known for his self-given name 'BTK' (Bind, Torture, Kill).",
      image: "images/criminal-profiles/btk.jpg",
      filePath: "gold-standards/btk-otero-profile.json"
    },
    {
      id: "ed-kemper",
      name: "Edmund Kemper",
      subtitle: "Co-ed Killer",
      description: "American serial killer who murdered ten people, including his paternal grandparents and mother, between 1964 and 1973.",
      image: "images/criminal-profiles/ed-kemper.jpg",
      filePath: "gold-standards/ed-kemper-profile.json"
    },
    {
      id: "mad-bomber",
      name: "Mad Bomber",
      subtitle: "George Metesky",
      description: "American domestic terrorist who planted bombs in New York City between 1940 and 1957, primarily targeting utilities and theaters.",
      image: "images/criminal-profiles/george-metesky.jpg",
      filePath: "gold-standards/mad-bomber-profile.json"
    },
    {
      id: "robert-napper-rachel-nickell",
      name: "Robert Napper",
      subtitle: "Rachel Nickell Murder",
      description: "British serial rapist and murderer who killed Rachel Nickell on Wimbledon Common in London in 1992.",
      image: "images/criminal-profiles/robert-napper.jpg",
      filePath: "gold-standards/robert-napper-rachel-nickell-profile.json"
    },
    {
      id: "unabomber",
      name: "Unabomber",
      subtitle: "Theodore Kaczynski",
      description: "American domestic terrorist and former mathematics professor who conducted a bombing campaign targeting people involved with modern technology.",
      image: "images/criminal-profiles/theodore-kaczynski.jpg",
      filePath: "gold-standards/unabomber-profile.json"
    }
  ],
  
  // Helper function to construct model file paths
  getModelFilePath: function(run, testCase, model) {
    return `generated_profiles/${run}/${testCase}/${model}_result.json`;
  }
};