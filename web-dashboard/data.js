// Forensic-LLM2 Evaluation Data
const evaluationData = [
  {
    "test_case": "ted-bundy-lake",
    "model": "Claude-3.7-Sonnet",
    "completeness": 1.0,
    "reasoning_count": 45,
    "avg_framework_confidence": 0.875,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.86,
      "sexual_behavioral_analysis": 0.97,
      "behavioral_change_staging": 0.905,
      "spatial_behavioral_analysis": 0.255
    },
    "framework_agreement": 0.6666666666666666,
    "semantic_similarity": 0.8563104271888733,
    "processing_time": 80.10090732574463
  },
  {
    "test_case": "ted-bundy-lake",
    "model": "Claude-3.7-Sonnet-Thinking",
    "completeness": 1.0,
    "reasoning_count": 45,
    "avg_framework_confidence": 0.825,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.925,
      "sexual_behavioral_analysis": 0.955,
      "behavioral_change_staging": 0.97,
      "spatial_behavioral_analysis": 0.94
    },
    "framework_agreement": 0.6666666666666666,
    "semantic_similarity": 0.8444085717201233,
    "processing_time": 103.52448391914368
  },
  {
    "test_case": "ted-bundy-lake",
    "model": "Llama-3.3-70B-Instruct",
    "completeness": 1.0,
    "reasoning_count": 16,
    "avg_framework_confidence": 0.575,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.47,
      "sexual_behavioral_analysis": 0.44,
      "behavioral_change_staging": 0.38,
      "spatial_behavioral_analysis": 0.44
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.8405808210372925,
    "processing_time": 59.41131639480591
  },
  {
    "test_case": "ted-bundy-lake",
    "model": "DeepSeek-R1",
    "completeness": 1.0,
    "reasoning_count": 22,
    "avg_framework_confidence": 0.7625,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.57,
      "sexual_behavioral_analysis": 0.635,
      "behavioral_change_staging": 0.53,
      "spatial_behavioral_analysis": 0.5
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.7943567037582397,
    "processing_time": 208.3317220211029
  },
  {
    "test_case": "ted-bundy-lake",
    "model": "GPT-4o",
    "completeness": 1.0,
    "reasoning_count": 17,
    "avg_framework_confidence": 0.825,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.55,
      "sexual_behavioral_analysis": 0.58,
      "behavioral_change_staging": 0.53,
      "spatial_behavioral_analysis": 0.53
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.8346458673477173,
    "processing_time": 63.889294385910034
  },
  {
    "test_case": "ted-bundy-lake",
    "model": "GPT-4o-mini",
    "completeness": 1.0,
    "reasoning_count": 15,
    "avg_framework_confidence": 0.825,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.595,
      "sexual_behavioral_analysis": 0.58,
      "behavioral_change_staging": 0.485,
      "spatial_behavioral_analysis": 0.53
    },
    "framework_agreement": 0.0,
    "semantic_similarity": 0.8140509724617004,
    "processing_time": 90.11775279045105
  },
  {
    "test_case": "ted-bundy-lake",
    "model": "Gemini-2.0-Flash",
    "completeness": 1.0,
    "reasoning_count": 30,
    "avg_framework_confidence": 0.725,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.735,
      "sexual_behavioral_analysis": 0.825,
      "behavioral_change_staging": 0.84,
      "spatial_behavioral_analysis": 0.75
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.8241640329360962,
    "processing_time": 15.82123875617981
  },
  {
    "test_case": "unabomber",
    "model": "Claude-3.7-Sonnet",
    "completeness": 1.0,
    "reasoning_count": 37,
    "avg_framework_confidence": 0.6125,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.66,
      "sexual_behavioral_analysis": 0.1,
      "behavioral_change_staging": 0.69,
      "spatial_behavioral_analysis": 0.645
    },
    "framework_agreement": 0.0,
    "semantic_similarity": 0.8346519470214844,
    "processing_time": 66.32826900482178
  },
  {
    "test_case": "unabomber",
    "model": "Claude-3.7-Sonnet-Thinking",
    "completeness": 1.0,
    "reasoning_count": 45,
    "avg_framework_confidence": 0.7875,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.86,
      "sexual_behavioral_analysis": 0.72,
      "behavioral_change_staging": 0.89,
      "spatial_behavioral_analysis": 0.795
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.819451093673706,
    "processing_time": 118.8128969669342
  },
  {
    "test_case": "btk-otero",
    "model": "Claude-3.7-Sonnet",
    "completeness": 1.0,
    "reasoning_count": 48,
    "avg_framework_confidence": 0.7875,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.91,
      "sexual_behavioral_analysis": 0.94,
      "behavioral_change_staging": 0.97,
      "spatial_behavioral_analysis": 0.845
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.8956945538520813,
    "processing_time": 61.33
  },
  {
    "test_case": "btk-otero",
    "model": "Llama-3.3-70B-Instruct",
    "completeness": 1.0,
    "reasoning_count": 22,
    "avg_framework_confidence": 0.65,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.57,
      "sexual_behavioral_analysis": 0.5,
      "behavioral_change_staging": 0.44,
      "spatial_behavioral_analysis": 0.33
    },
    "framework_agreement": 0.0,
    "semantic_similarity": 0.8853046298027039,
    "processing_time": 27.739761114120483
  },
  {
    "test_case": "mad-bomber",
    "model": "GPT-4o",
    "completeness": 1.0,
    "reasoning_count": 32,
    "avg_framework_confidence": 0.7,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.61,
      "sexual_behavioral_analysis": 0.06,
      "behavioral_change_staging": 0.5,
      "spatial_behavioral_analysis": 0.45
    },
    "framework_agreement": 0.3333333333333333,
    "semantic_similarity": 0.8574005961418152,
    "processing_time": 76.34671211242676
  },
  {
    "test_case": "mad-bomber",
    "model": "Gemini-2.0-Flash",
    "completeness": 1.0,
    "reasoning_count": 37,
    "avg_framework_confidence": 0.8,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.71,
      "sexual_behavioral_analysis": 0.0,
      "behavioral_change_staging": 0.76,
      "spatial_behavioral_analysis": 0.57
    },
    "framework_agreement": 0.0,
    "semantic_similarity": 0.8474889397621155,
    "processing_time": 18.12981390953064
  },
  {
    "test_case": "ed-kemper",
    "model": "Claude-3.7-Sonnet",
    "completeness": 1.0,
    "reasoning_count": 46,
    "avg_framework_confidence": 0.825,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.94,
      "sexual_behavioral_analysis": 0.97,
      "behavioral_change_staging": 0.875,
      "spatial_behavioral_analysis": 0.845
    },
    "framework_agreement": 0.0,
    "semantic_similarity": 0.7948462963104248,
    "processing_time": 57.29
  },
  {
    "test_case": "ed-kemper",
    "model": "DeepSeek-R1",
    "completeness": 1.0,
    "reasoning_count": 27,
    "avg_framework_confidence": 0.775,
    "consistency": 1.0,
    "framework_contributions": {
      "narrative_action_system": 0.65,
      "sexual_behavioral_analysis": 0.76,
      "behavioral_change_staging": 0.695,
      "spatial_behavioral_analysis": 0.665
    },
    "framework_agreement": 0.0,
    "semantic_similarity": 0.7835459113121033,
    "processing_time": 144.6731367111206
  }
];

// Helper functions for data manipulation

// Get unique test cases
function getUniqueTestCases() {
  return [...new Set(evaluationData.map(item => item.test_case))];
}

// Get unique models
function getUniqueModels() {
  return [...new Set(evaluationData.map(item => item.model))];
}

// Get average metrics across all models
function getAverageMetrics() {
  const metricSums = {
    completeness: 0,
    reasoning_count: 0,
    avg_framework_confidence: 0,
    framework_agreement: 0,
    semantic_similarity: 0
  };
  
  evaluationData.forEach(item => {
    metricSums.completeness += item.completeness;
    metricSums.reasoning_count += item.reasoning_count;
    metricSums.avg_framework_confidence += item.avg_framework_confidence;
    metricSums.framework_agreement += item.framework_agreement;
    metricSums.semantic_similarity += item.semantic_similarity;
  });
  
  const count = evaluationData.length;
  
  return {
    completeness: (metricSums.completeness / count).toFixed(2),
    reasoning_count: (metricSums.reasoning_count / count).toFixed(1),
    avg_framework_confidence: (metricSums.avg_framework_confidence / count).toFixed(2),
    framework_agreement: (metricSums.framework_agreement / count).toFixed(2),
    semantic_similarity: (metricSums.semantic_similarity / count).toFixed(2)
  };
}

// Get model performance summary by aggregating across test cases
function getModelPerformanceSummary() {
  const models = getUniqueModels();
  const summary = {};
  
  models.forEach(model => {
    const modelData = evaluationData.filter(item => item.model === model);
    
    const metricSums = {
      completeness: 0,
      reasoning_count: 0,
      avg_framework_confidence: 0,
      framework_agreement: 0,
      semantic_similarity: 0,
      processing_time: 0,
      framework_contributions: {
        narrative_action_system: 0,
        sexual_behavioral_analysis: 0,
        behavioral_change_staging: 0,
        spatial_behavioral_analysis: 0
      }
    };
    
    modelData.forEach(item => {
      metricSums.completeness += item.completeness;
      metricSums.reasoning_count += item.reasoning_count;
      metricSums.avg_framework_confidence += item.avg_framework_confidence;
      metricSums.framework_agreement += item.framework_agreement;
      metricSums.semantic_similarity += item.semantic_similarity;
      metricSums.processing_time += item.processing_time;
      
      if (item.framework_contributions) {
        metricSums.framework_contributions.narrative_action_system += item.framework_contributions.narrative_action_system || 0;
        metricSums.framework_contributions.sexual_behavioral_analysis += item.framework_contributions.sexual_behavioral_analysis || 0;
        metricSums.framework_contributions.behavioral_change_staging += item.framework_contributions.behavioral_change_staging || 0;
        metricSums.framework_contributions.spatial_behavioral_analysis += item.framework_contributions.spatial_behavioral_analysis || 0;
      }
    });
    
    const count = modelData.length;
    
    summary[model] = {
      completeness: (metricSums.completeness / count).toFixed(2),
      reasoning_count: (metricSums.reasoning_count / count).toFixed(1),
      avg_framework_confidence: (metricSums.avg_framework_confidence / count).toFixed(2),
      framework_agreement: (metricSums.framework_agreement / count).toFixed(2),
      semantic_similarity: (metricSums.semantic_similarity / count).toFixed(2),
      processing_time: (metricSums.processing_time / count).toFixed(2),
      framework_contributions: {
        narrative_action_system: (metricSums.framework_contributions.narrative_action_system / count).toFixed(2),
        sexual_behavioral_analysis: (metricSums.framework_contributions.sexual_behavioral_analysis / count).toFixed(2),
        behavioral_change_staging: (metricSums.framework_contributions.behavioral_change_staging / count).toFixed(2),
        spatial_behavioral_analysis: (metricSums.framework_contributions.spatial_behavioral_analysis / count).toFixed(2)
      }
    };
  });
  
  return summary;
}

// Filter data by test case
function filterByTestCase(testCase) {
  if (testCase === 'all') {
    return evaluationData;
  }
  return evaluationData.filter(item => item.test_case === testCase);
}

// Filter data by models
function filterByModels(selectedModels) {
  return evaluationData.filter(item => selectedModels.includes(item.model));
}

// Color palette for charts
const colorPalette = [
  'rgba(52, 152, 219, 0.7)',  // Blue
  'rgba(231, 76, 60, 0.7)',   // Red
  'rgba(46, 204, 113, 0.7)',  // Green
  'rgba(155, 89, 182, 0.7)',  // Purple
  'rgba(243, 156, 18, 0.7)',  // Orange
  'rgba(26, 188, 156, 0.7)',  // Turquoise
  'rgba(52, 73, 94, 0.7)',    // Dark Blue
  'rgba(211, 84, 0, 0.7)'     // Dark Orange
];

// Format number for display
function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
} 