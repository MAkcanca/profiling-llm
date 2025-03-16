# Forensic LLM Analysis Research Report

*Generated on 2025-03-16*

## Executive Summary

This report contains detailed statistics and analyses of different language models' performance on forensic analysis tasks. The data is structured to facilitate the writing of a scientific paper based on these results.

### Methodological Notes

#### Statistical Methodology

- **Multiple Comparisons Correction**: False Discovery Rate (FDR) correction using the Benjamini-Hochberg procedure has been applied to control the expected proportion of false positives in all pairwise comparisons. This procedure adjusts p-values to control the proportion of Type I errors (false positives) when conducting multiple statistical tests, offering greater statistical power than family-wise error rate methods.

#### Limitations

- **Sample Size**: This study analyzed a limited set of test cases. Results should be interpreted as preliminary findings that warrant further investigation with a larger dataset.

- **Framework Specificity**: Performance metrics may be specific to the forensic frameworks used in this evaluation and may not generalize to all forensic analysis tasks.

## Dataset Overview

- **Total test cases evaluated**: 6
- **Number of models evaluated**: 12
- **Models included in analysis**: o3-mini, Gemma-3, GPT-4.5-Preview, o3-mini-high, GPT-4o, GPT-4o-mini, Claude-3.7-Sonnet, Claude-3.7-Sonnet-Thinking, Llama-3.3-70B-Instruct, DeepSeek-R1, Gemini-2.0-Flash, Gemini-2.0-Flash-Thinking-Exp0121

## Model Performance Summary

### Overall Performance Metrics

| Model | Accuracy: Narrative Action System | Accuracy: Spatial Behavioral Analysis | Accuracy: Sexual Behavioral Analysis | Accuracy: Sexual Homicide Pathways Analysis | Average |
|-------|-----------------------------------|---------------------------------------|--------------------------------------|----------------------------------------------|-------|
| Claude-3.7-Sonnet | 0.667 | 0.500 | 0.500 | 0.750 | 0.604 |
| Claude-3.7-Sonnet-Thinking | 0.625 | 0.458 | 0.500 | 0.750 | 0.583 |
| DeepSeek-R1 | 0.625 | 0.375 | 0.438 | 0.750 | 0.547 |
| o3-mini | 0.708 | 0.583 | 0.062 | 0.750 | 0.526 |
| Llama-3.3-70B-Instruct | 0.708 | 0.458 | 0.125 | 0.750 | 0.510 |
| GPT-4o | 0.583 | 0.417 | 0.250 | 0.688 | 0.484 |
| GPT-4o-mini | 0.500 | 0.750 | 0.125 | 0.562 | 0.484 |
| o3-mini-high | 0.667 | 0.500 | 0.000 | 0.750 | 0.479 |
| GPT-4.5-Preview | 0.375 | 0.583 | 0.312 | 0.625 | 0.474 |
| Gemma-3 | 0.458 | 0.708 | 0.000 | 0.625 | 0.448 |
| Gemini-2.0-Flash | 0.750 | 0.458 | 0.125 | 0.438 | 0.443 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | 0.333 | 0.188 | 0.625 | 0.370 |

## Framework-Specific Analysis

## Framework Contribution Analysis

This section analyzes how much each framework contributes to the overall accuracy of profiles.

### Individual Framework Contributions

This table shows how much each framework contributes to the overall accuracy. Positive values indicate correct classifications that improve overall accuracy, while negative values represent potential contributions if corrected.

| Framework | Average Contribution | Absolute Contribution |
|-----------|----------------------|----------------------|
| Spatial Behavioral Analysis | 0.3333 | 0.3333 |
| Narrative Action System | 0.3333 | 0.3333 |
| Sexual Homicide Pathways Analysis | 0.2500 | 0.2500 |
| Sexual Behavioral Analysis | 0.2500 | 0.2500 |

### Relative Framework Importance

This table shows the relative importance of each framework (normalized contribution). Higher values indicate frameworks that have a greater impact on overall profile accuracy.

| Framework | Relative Importance |
|-----------|---------------------|
| Spatial Behavioral Analysis | 0.3333 |
| Narrative Action System | 0.3333 |
| Sexual Homicide Pathways Analysis | 0.2500 |
| Sexual Behavioral Analysis | 0.2500 |

### Most Influential Frameworks

This analysis identifies which frameworks are most frequently identified as the most influential for overall profile accuracy.

| Framework | Count | % of Cases |
|-----------|-------|------------|
| Narrative Action System | 288 | 100.0% |
| Spatial Behavioral Analysis | 288 | 100.0% |
| Sexual Behavioral Analysis | 192 | 66.7% |

### Spatial Behavioral Analysis Framework

#### Accuracy Analysis for Spatial Behavioral Analysis

| Model | Accuracy | Sample Count |
|-------|----------|-------------|
| GPT-4o-mini | 0.750 | 24 |
| Gemma-3 | 0.708 | 24 |
| GPT-4.5-Preview | 0.583 | 24 |
| o3-mini | 0.583 | 24 |
| Claude-3.7-Sonnet | 0.500 | 24 |
| o3-mini-high | 0.500 | 24 |
| Gemini-2.0-Flash | 0.458 | 24 |
| Claude-3.7-Sonnet-Thinking | 0.458 | 24 |
| Llama-3.3-70B-Instruct | 0.458 | 24 |
| GPT-4o | 0.417 | 24 |
| DeepSeek-R1 | 0.375 | 24 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | 24 |

#### Confidence Analysis for Spatial Behavioral Analysis

| Model | Confidence |
|-------|------------|
| GPT-4.5-Preview | 0.881 |
| GPT-4o-mini | 0.858 |
| GPT-4o | 0.844 |
| o3-mini-high | 0.825 |
| Gemma-3 | 0.802 |
| o3-mini | 0.800 |
| Claude-3.7-Sonnet-Thinking | 0.798 |
| Claude-3.7-Sonnet | 0.779 |
| Gemini-2.0-Flash | 0.771 |
| Llama-3.3-70B-Instruct | 0.760 |
| DeepSeek-R1 | 0.754 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.715 |

#### Completeness Analysis for Spatial Behavioral Analysis

| Model | Completeness |
|-------|-------------|
| Claude-3.7-Sonnet-Thinking | 1.000 |
| DeepSeek-R1 | 1.000 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 1.000 |
| o3-mini-high | 0.983 |
| o3-mini | 0.983 |
| Claude-3.7-Sonnet | 0.967 |
| Llama-3.3-70B-Instruct | 0.958 |
| GPT-4o | 0.958 |
| Gemma-3 | 0.917 |
| GPT-4.5-Preview | 0.850 |
| Gemini-2.0-Flash | 0.842 |
| GPT-4o-mini | 0.833 |

### Sexual Homicide Pathways Analysis Framework

#### Accuracy Analysis for Sexual Homicide Pathways Analysis

| Model | Accuracy | Sample Count |
|-------|----------|-------------|
| Claude-3.7-Sonnet | 0.750 | 16 |
| Claude-3.7-Sonnet-Thinking | 0.750 | 16 |
| DeepSeek-R1 | 0.750 | 16 |
| o3-mini | 0.750 | 16 |
| Llama-3.3-70B-Instruct | 0.750 | 16 |
| o3-mini-high | 0.750 | 16 |
| GPT-4o | 0.688 | 16 |
| GPT-4.5-Preview | 0.625 | 16 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.625 | 16 |
| Gemma-3 | 0.625 | 16 |
| GPT-4o-mini | 0.562 | 16 |
| Gemini-2.0-Flash | 0.438 | 16 |

#### Confidence Analysis for Sexual Homicide Pathways Analysis

| Model | Confidence |
|-------|------------|
| Gemma-3 | 0.758 |
| DeepSeek-R1 | 0.612 |
| GPT-4o | 0.604 |
| GPT-4.5-Preview | 0.599 |
| GPT-4o-mini | 0.573 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.567 |
| Claude-3.7-Sonnet-Thinking | 0.554 |
| Claude-3.7-Sonnet | 0.550 |
| o3-mini-high | 0.548 |
| Llama-3.3-70B-Instruct | 0.529 |
| o3-mini | 0.525 |
| Gemini-2.0-Flash | 0.448 |

#### Completeness Analysis for Sexual Homicide Pathways Analysis

| Model | Completeness |
|-------|-------------|
| Gemma-3 | 0.817 |
| DeepSeek-R1 | 0.792 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.750 |
| GPT-4o | 0.700 |
| Llama-3.3-70B-Instruct | 0.692 |
| Claude-3.7-Sonnet | 0.667 |
| Claude-3.7-Sonnet-Thinking | 0.667 |
| o3-mini | 0.658 |
| o3-mini-high | 0.642 |
| Gemini-2.0-Flash | 0.567 |
| GPT-4.5-Preview | 0.550 |
| GPT-4o-mini | 0.533 |

### Narrative Action System Framework

#### Accuracy Analysis for Narrative Action System

| Model | Accuracy | Sample Count |
|-------|----------|-------------|
| Gemini-2.0-Flash | 0.750 | 24 |
| Llama-3.3-70B-Instruct | 0.708 | 24 |
| o3-mini | 0.708 | 24 |
| Claude-3.7-Sonnet | 0.667 | 24 |
| o3-mini-high | 0.667 | 24 |
| Claude-3.7-Sonnet-Thinking | 0.625 | 24 |
| DeepSeek-R1 | 0.625 | 24 |
| GPT-4o | 0.583 | 24 |
| GPT-4o-mini | 0.500 | 24 |
| Gemma-3 | 0.458 | 24 |
| GPT-4.5-Preview | 0.375 | 24 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | 24 |

#### Confidence Analysis for Narrative Action System

| Model | Confidence |
|-------|------------|
| GPT-4.5-Preview | 0.877 |
| GPT-4o | 0.856 |
| GPT-4o-mini | 0.850 |
| o3-mini | 0.846 |
| o3-mini-high | 0.844 |
| Claude-3.7-Sonnet-Thinking | 0.842 |
| Claude-3.7-Sonnet | 0.829 |
| DeepSeek-R1 | 0.828 |
| Llama-3.3-70B-Instruct | 0.758 |
| Gemma-3 | 0.758 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.727 |
| Gemini-2.0-Flash | 0.690 |

#### Completeness Analysis for Narrative Action System

| Model | Completeness |
|-------|-------------|
| Claude-3.7-Sonnet | 1.000 |
| Claude-3.7-Sonnet-Thinking | 1.000 |
| DeepSeek-R1 | 1.000 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.992 |
| o3-mini | 0.992 |
| o3-mini-high | 0.992 |
| GPT-4o | 0.975 |
| Llama-3.3-70B-Instruct | 0.958 |
| Gemma-3 | 0.950 |
| Gemini-2.0-Flash | 0.892 |
| GPT-4.5-Preview | 0.883 |
| GPT-4o-mini | 0.850 |

### Sexual Behavioral Analysis Framework

#### Accuracy Analysis for Sexual Behavioral Analysis

| Model | Accuracy | Sample Count |
|-------|----------|-------------|
| Claude-3.7-Sonnet | 0.500 | 16 |
| Claude-3.7-Sonnet-Thinking | 0.500 | 16 |
| DeepSeek-R1 | 0.438 | 16 |
| GPT-4.5-Preview | 0.312 | 16 |
| GPT-4o | 0.250 | 16 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.188 | 16 |
| GPT-4o-mini | 0.125 | 16 |
| Gemini-2.0-Flash | 0.125 | 16 |
| Llama-3.3-70B-Instruct | 0.125 | 16 |
| o3-mini | 0.062 | 16 |
| Gemma-3 | 0.000 | 16 |
| o3-mini-high | 0.000 | 16 |

#### Confidence Analysis for Sexual Behavioral Analysis

| Model | Confidence |
|-------|------------|
| GPT-4o-mini | 0.685 |
| Gemma-3 | 0.662 |
| GPT-4.5-Preview | 0.606 |
| Claude-3.7-Sonnet-Thinking | 0.571 |
| Claude-3.7-Sonnet | 0.558 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.544 |
| DeepSeek-R1 | 0.514 |
| GPT-4o | 0.513 |
| o3-mini-high | 0.490 |
| Llama-3.3-70B-Instruct | 0.471 |
| Gemini-2.0-Flash | 0.454 |
| o3-mini | 0.417 |

#### Completeness Analysis for Sexual Behavioral Analysis

| Model | Completeness |
|-------|-------------|
| Gemma-3 | 0.808 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.708 |
| Claude-3.7-Sonnet | 0.667 |
| Claude-3.7-Sonnet-Thinking | 0.667 |
| GPT-4o-mini | 0.667 |
| DeepSeek-R1 | 0.667 |
| Llama-3.3-70B-Instruct | 0.650 |
| GPT-4o | 0.633 |
| o3-mini-high | 0.600 |
| Gemini-2.0-Flash | 0.600 |
| o3-mini | 0.567 |
| GPT-4.5-Preview | 0.550 |

## Reasoning Analysis

### Detailed Reasoning by Section

| Model | Overall | Demographics | Psychological Characteristics | Behavioral Characteristics | Geographic Behavior | Skills And Knowledge | Investigative Implications | Key Identifiers | Framework Narrative Action System | Framework Sexual Behavioral Analysis | Framework Spatial Behavioral Analysis | Framework Sexual Homicide Pathways Analysis | Frameworks Total | Total |
|-------|---------|--------------|-------------------------------|----------------------------|---------------------|----------------------|----------------------------|-----------------|-----------------------------------|--------------------------------------|---------------------------------------|---------------------------------------------|------------------|-------|
| Claude-3.7-Sonnet | 7.33 | 5.71 | 5.96 | 6.04 | 3.75 | 5.17 | 5.21 | 5.21 | 3.38 | 2.46 | 3.46 | 2.54 | 11.83 | 68.04 |
| Claude-3.7-Sonnet-Thinking | 6.75 | 5.71 | 5.79 | 5.79 | 3.46 | 5.04 | 5.04 | 5.08 | 3.25 | 2.42 | 3.83 | 2.42 | 11.92 | 66.50 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 4.12 | 5.58 | 5.83 | 5.96 | 3.00 | 4.83 | 5.00 | 4.96 | 1.04 | 0.71 | 1.04 | 0.75 | 3.54 | 46.37 |
| Gemini-2.0-Flash | 4.62 | 4.46 | 4.62 | 4.71 | 2.71 | 3.75 | 3.92 | 3.71 | 1.67 | 1.04 | 1.50 | 0.83 | 5.04 | 42.58 |
| DeepSeek-R1 | 4.29 | 3.46 | 3.25 | 3.25 | 2.96 | 3.08 | 3.08 | 3.08 | 2.12 | 1.21 | 2.17 | 1.46 | 6.96 | 40.38 |
| Gemma-3 | 3.54 | 3.54 | 3.67 | 3.42 | 2.42 | 2.83 | 3.00 | 2.88 | 1.08 | 0.92 | 1.08 | 0.96 | 4.04 | 33.37 |
| GPT-4o | 2.79 | 3.83 | 3.71 | 3.54 | 2.25 | 2.67 | 2.88 | 2.71 | 1.25 | 0.79 | 1.21 | 0.83 | 4.08 | 32.54 |
| GPT-4o-mini | 2.92 | 2.79 | 2.50 | 2.33 | 2.17 | 2.21 | 2.12 | 2.17 | 1.71 | 1.21 | 1.42 | 0.83 | 5.17 | 29.54 |
| o3-mini-high | 3.79 | 2.54 | 2.25 | 2.21 | 2.12 | 2.12 | 2.12 | 2.12 | 1.00 | 0.62 | 1.00 | 0.67 | 3.29 | 25.88 |
| o3-mini | 3.75 | 2.50 | 2.25 | 2.21 | 2.08 | 2.12 | 2.12 | 2.12 | 1.00 | 0.58 | 1.00 | 0.67 | 3.25 | 25.67 |
| Llama-3.3-70B-Instruct | 2.12 | 2.29 | 2.17 | 2.12 | 1.88 | 1.92 | 2.00 | 1.83 | 1.42 | 0.83 | 1.42 | 0.92 | 4.58 | 25.50 |
| GPT-4.5-Preview | 1.12 | 2.83 | 2.38 | 2.17 | 1.88 | 1.71 | 1.71 | 1.71 | 1.00 | 0.67 | 1.00 | 0.67 | 3.33 | 22.17 |

## Correlation Analysis

### Correlation Matrix Between Key Metrics

|                                        |   Avg Framework Conf |   Acc: Narrative Action System |   Acc: Spatial Behavioral Analysis |   Acc: Sexual Behavioral Analysis |   Acc: Sexual Homicide Pathways Analysis |
|:---------------------------------------|---------------------:|-------------------------------:|-----------------------------------:|----------------------------------:|-----------------------------------------:|
| Avg Framework Conf                     |                 1    |                          -0.16 |                               0.05 |                              0.02 |                                    -0.2  |
| Acc: Narrative Action System           |                -0.16 |                           1    |                               0.19 |                              0.14 |                                     0.3  |
| Acc: Spatial Behavioral Analysis       |                 0.05 |                           0.19 |                               1    |                              0.14 |                                     0.08 |
| Acc: Sexual Behavioral Analysis        |                 0.02 |                           0.14 |                               0.14 |                              1    |                                     0.32 |
| Acc: Sexual Homicide Pathways Analysis |                -0.2  |                           0.3  |                               0.08 |                              0.32 |                                     1    |

*Note: Correlation values range from -1 (perfect negative correlation) to 1 (perfect positive correlation). 0 indicates no correlation.*

## Statistical Significance Analysis

### Accuracy: Narrative Action System

*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*

#### Repeated Measures ANOVA

Repeated Measures ANOVA test for differences in accuracy: narrative action system between models (accounting for test case variability):

- F-value: 1.3348
- p-value: 0.2308
- Degrees of freedom: 11, 55

The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a statistically significant difference in accuracy: narrative action system between models when controlling for test case variability.

This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, which provides a more accurate assessment than treating each test case as independent.

#### Descriptive Statistics by Model

| Model | Mean | Std | Min | Max | Count | 95% CI |
|-------|------|-----|-----|-----|-------|---------|
| Gemini-2.0-Flash | 0.750 | 0.442 | 0.000 | 1.000 | 24 | [0.563, 0.937] |
| o3-mini | 0.708 | 0.464 | 0.000 | 1.000 | 24 | [0.512, 0.904] |
| Llama-3.3-70B-Instruct | 0.708 | 0.464 | 0.000 | 1.000 | 24 | [0.512, 0.904] |
| o3-mini-high | 0.667 | 0.482 | 0.000 | 1.000 | 24 | [0.463, 0.870] |
| Claude-3.7-Sonnet | 0.667 | 0.482 | 0.000 | 1.000 | 24 | [0.463, 0.870] |
| Claude-3.7-Sonnet-Thinking | 0.625 | 0.495 | 0.000 | 1.000 | 24 | [0.416, 0.834] |
| DeepSeek-R1 | 0.625 | 0.495 | 0.000 | 1.000 | 24 | [0.416, 0.834] |
| GPT-4o | 0.583 | 0.504 | 0.000 | 1.000 | 24 | [0.371, 0.796] |
| GPT-4o-mini | 0.500 | 0.511 | 0.000 | 1.000 | 24 | [0.284, 0.716] |
| Gemma-3 | 0.458 | 0.509 | 0.000 | 1.000 | 24 | [0.243, 0.673] |
| GPT-4.5-Preview | 0.375 | 0.495 | 0.000 | 1.000 | 24 | [0.166, 0.584] |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | 0.482 | 0.000 | 1.000 | 24 | [0.130, 0.537] |

#### Effect Size Analysis (Cohen's d)

Cohen's d measures the standardized difference between two means. It indicates the magnitude of the effect:

- Small effect: d < 0.5
- Medium effect: 0.5 ≤ d < 0.8
- Large effect: d ≥ 0.8

**Note:** Benjamini-Hochberg False Discovery Rate (FDR) correction has been applied to control for 66 multiple comparisons.
When multiple statistical tests are performed, the probability of observing a significant result by chance increases.
The Benjamini-Hochberg procedure adjusts p-values to control the expected proportion of false discoveries among all rejected hypotheses.
Effect sizes labeled with '(corrected)' indicate comparisons that were not statistically significant after correction.

| Model 1 | Model 2 | Cohen's d | Effect Size |
|---------|---------|-----------|-------------|
| Gemini-2.0-Flash | Gemini-2.0-Flash-Thinking-Exp0121 | 0.901 | Medium (corrected) |
| GPT-4.5-Preview | Gemini-2.0-Flash | 0.799 | Small (corrected) |
| o3-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.793 | Small (corrected) |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash-Thinking-Exp0121 | 0.793 | Small (corrected) |
| o3-mini | GPT-4.5-Preview | 0.695 | Small (corrected) |
| GPT-4.5-Preview | Llama-3.3-70B-Instruct | 0.695 | Small (corrected) |
| o3-mini-high | Gemini-2.0-Flash-Thinking-Exp0121 | 0.692 | Small (corrected) |
| Claude-3.7-Sonnet | Gemini-2.0-Flash-Thinking-Exp0121 | 0.692 | Small (corrected) |
| Gemma-3 | Gemini-2.0-Flash | 0.612 | Small (corrected) |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash-Thinking-Exp0121 | 0.598 | Small (corrected) |
| DeepSeek-R1 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.598 | Small (corrected) |
| GPT-4.5-Preview | o3-mini-high | 0.598 | Small (corrected) |
| GPT-4.5-Preview | Claude-3.7-Sonnet | 0.598 | Small (corrected) |
| GPT-4o-mini | Gemini-2.0-Flash | 0.523 | Small (corrected) |
| o3-mini | Gemma-3 | 0.513 | Small (corrected) |
| Gemma-3 | Llama-3.3-70B-Instruct | 0.513 | Small (corrected) |
| GPT-4o | Gemini-2.0-Flash-Thinking-Exp0121 | 0.507 | Small (corrected) |
| GPT-4.5-Preview | Claude-3.7-Sonnet-Thinking | 0.506 | Small (corrected) |
| GPT-4.5-Preview | DeepSeek-R1 | 0.506 | Small (corrected) |
| o3-mini | GPT-4o-mini | 0.427 | Small |
| GPT-4o-mini | Llama-3.3-70B-Instruct | 0.427 | Small |
| Gemma-3 | o3-mini-high | 0.420 | Small |
| Gemma-3 | Claude-3.7-Sonnet | 0.420 | Small |
| GPT-4.5-Preview | GPT-4o | 0.417 | Small |
| GPT-4o | Gemini-2.0-Flash | 0.352 | Small |
| o3-mini-high | GPT-4o-mini | 0.336 | Small |
| GPT-4o-mini | Claude-3.7-Sonnet | 0.336 | Small |
| GPT-4o-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.336 | Small |
| Gemma-3 | Claude-3.7-Sonnet-Thinking | 0.332 | Small |
| Gemma-3 | DeepSeek-R1 | 0.332 | Small |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash | 0.266 | Small |
| DeepSeek-R1 | Gemini-2.0-Flash | 0.266 | Small |
| o3-mini | GPT-4o | 0.258 | Small |
| GPT-4o | Llama-3.3-70B-Instruct | 0.258 | Small |
| Gemma-3 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.252 | Small |
| GPT-4.5-Preview | GPT-4o-mini | 0.249 | Small |
| GPT-4o-mini | Claude-3.7-Sonnet-Thinking | 0.249 | Small |
| GPT-4o-mini | DeepSeek-R1 | 0.249 | Small |
| Gemma-3 | GPT-4o | 0.247 | Small |
| o3-mini-high | Gemini-2.0-Flash | 0.180 | Small |
| Claude-3.7-Sonnet | Gemini-2.0-Flash | 0.180 | Small |
| o3-mini | Claude-3.7-Sonnet-Thinking | 0.174 | Small |
| o3-mini | DeepSeek-R1 | 0.174 | Small |
| Claude-3.7-Sonnet-Thinking | Llama-3.3-70B-Instruct | 0.174 | Small |
| Llama-3.3-70B-Instruct | DeepSeek-R1 | 0.174 | Small |
| o3-mini-high | GPT-4o | 0.169 | Small |
| GPT-4o | Claude-3.7-Sonnet | 0.169 | Small |
| Gemma-3 | GPT-4.5-Preview | 0.166 | Small |
| GPT-4o | GPT-4o-mini | 0.164 | Small |
| o3-mini | Gemini-2.0-Flash | 0.092 | Small |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash | 0.092 | Small |
| o3-mini | o3-mini-high | 0.088 | Small |
| o3-mini | Claude-3.7-Sonnet | 0.088 | Small |
| o3-mini-high | Llama-3.3-70B-Instruct | 0.088 | Small |
| Claude-3.7-Sonnet | Llama-3.3-70B-Instruct | 0.088 | Small |
| GPT-4.5-Preview | Gemini-2.0-Flash-Thinking-Exp0121 | 0.085 | Small |
| o3-mini-high | Claude-3.7-Sonnet-Thinking | 0.085 | Small |
| o3-mini-high | DeepSeek-R1 | 0.085 | Small |
| Claude-3.7-Sonnet | Claude-3.7-Sonnet-Thinking | 0.085 | Small |
| Claude-3.7-Sonnet | DeepSeek-R1 | 0.085 | Small |
| GPT-4o | Claude-3.7-Sonnet-Thinking | 0.083 | Small |
| GPT-4o | DeepSeek-R1 | 0.083 | Small |
| Gemma-3 | GPT-4o-mini | 0.082 | Small |
| o3-mini | Llama-3.3-70B-Instruct | 0.000 | Small |
| o3-mini-high | Claude-3.7-Sonnet | 0.000 | Small |
| Claude-3.7-Sonnet-Thinking | DeepSeek-R1 | 0.000 | Small |

### Accuracy: Spatial Behavioral Analysis

*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*

#### Repeated Measures ANOVA

Repeated Measures ANOVA test for differences in accuracy: spatial behavioral analysis between models (accounting for test case variability):

- F-value: 1.0797
- p-value: 0.3942
- Degrees of freedom: 11, 55

The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a statistically significant difference in accuracy: spatial behavioral analysis between models when controlling for test case variability.

This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, which provides a more accurate assessment than treating each test case as independent.

#### Descriptive Statistics by Model

| Model | Mean | Std | Min | Max | Count | 95% CI |
|-------|------|-----|-----|-----|-------|---------|
| GPT-4o-mini | 0.750 | 0.442 | 0.000 | 1.000 | 24 | [0.563, 0.937] |
| Gemma-3 | 0.708 | 0.464 | 0.000 | 1.000 | 24 | [0.512, 0.904] |
| o3-mini | 0.583 | 0.504 | 0.000 | 1.000 | 24 | [0.371, 0.796] |
| GPT-4.5-Preview | 0.583 | 0.504 | 0.000 | 1.000 | 24 | [0.371, 0.796] |
| o3-mini-high | 0.500 | 0.511 | 0.000 | 1.000 | 24 | [0.284, 0.716] |
| Claude-3.7-Sonnet | 0.500 | 0.511 | 0.000 | 1.000 | 24 | [0.284, 0.716] |
| Claude-3.7-Sonnet-Thinking | 0.458 | 0.509 | 0.000 | 1.000 | 24 | [0.243, 0.673] |
| Llama-3.3-70B-Instruct | 0.458 | 0.509 | 0.000 | 1.000 | 24 | [0.243, 0.673] |
| Gemini-2.0-Flash | 0.458 | 0.509 | 0.000 | 1.000 | 24 | [0.243, 0.673] |
| GPT-4o | 0.417 | 0.504 | 0.000 | 1.000 | 24 | [0.204, 0.629] |
| DeepSeek-R1 | 0.375 | 0.495 | 0.000 | 1.000 | 24 | [0.166, 0.584] |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | 0.482 | 0.000 | 1.000 | 24 | [0.130, 0.537] |

#### Effect Size Analysis (Cohen's d)

Cohen's d measures the standardized difference between two means. It indicates the magnitude of the effect:

- Small effect: d < 0.5
- Medium effect: 0.5 ≤ d < 0.8
- Large effect: d ≥ 0.8

**Note:** Benjamini-Hochberg False Discovery Rate (FDR) correction has been applied to control for 66 multiple comparisons.
When multiple statistical tests are performed, the probability of observing a significant result by chance increases.
The Benjamini-Hochberg procedure adjusts p-values to control the expected proportion of false discoveries among all rejected hypotheses.
Effect sizes labeled with '(corrected)' indicate comparisons that were not statistically significant after correction.

| Model 1 | Model 2 | Cohen's d | Effect Size |
|---------|---------|-----------|-------------|
| GPT-4o-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.901 | Medium (corrected) |
| GPT-4o-mini | DeepSeek-R1 | 0.799 | Small (corrected) |
| Gemma-3 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.793 | Small (corrected) |
| GPT-4o | GPT-4o-mini | 0.703 | Small (corrected) |
| Gemma-3 | DeepSeek-R1 | 0.695 | Small (corrected) |
| GPT-4o-mini | Claude-3.7-Sonnet-Thinking | 0.612 | Small (corrected) |
| GPT-4o-mini | Llama-3.3-70B-Instruct | 0.612 | Small (corrected) |
| GPT-4o-mini | Gemini-2.0-Flash | 0.612 | Small (corrected) |
| Gemma-3 | GPT-4o | 0.602 | Small (corrected) |
| o3-mini-high | GPT-4o-mini | 0.523 | Small (corrected) |
| GPT-4o-mini | Claude-3.7-Sonnet | 0.523 | Small (corrected) |
| Gemma-3 | Claude-3.7-Sonnet-Thinking | 0.513 | Small (corrected) |
| Gemma-3 | Llama-3.3-70B-Instruct | 0.513 | Small (corrected) |
| Gemma-3 | Gemini-2.0-Flash | 0.513 | Small (corrected) |
| o3-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.507 | Small (corrected) |
| GPT-4.5-Preview | Gemini-2.0-Flash-Thinking-Exp0121 | 0.507 | Small (corrected) |
| Gemma-3 | o3-mini-high | 0.427 | Small |
| Gemma-3 | Claude-3.7-Sonnet | 0.427 | Small |
| o3-mini | DeepSeek-R1 | 0.417 | Small |
| GPT-4.5-Preview | DeepSeek-R1 | 0.417 | Small |
| o3-mini | GPT-4o-mini | 0.352 | Small |
| GPT-4.5-Preview | GPT-4o-mini | 0.352 | Small |
| o3-mini-high | Gemini-2.0-Flash-Thinking-Exp0121 | 0.336 | Small |
| Claude-3.7-Sonnet | Gemini-2.0-Flash-Thinking-Exp0121 | 0.336 | Small |
| o3-mini | GPT-4o | 0.331 | Small |
| GPT-4.5-Preview | GPT-4o | 0.331 | Small |
| o3-mini | Gemma-3 | 0.258 | Small |
| Gemma-3 | GPT-4.5-Preview | 0.258 | Small |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash-Thinking-Exp0121 | 0.252 | Small |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash-Thinking-Exp0121 | 0.252 | Small |
| Gemini-2.0-Flash | Gemini-2.0-Flash-Thinking-Exp0121 | 0.252 | Small |
| o3-mini-high | DeepSeek-R1 | 0.249 | Small |
| Claude-3.7-Sonnet | DeepSeek-R1 | 0.249 | Small |
| o3-mini | Claude-3.7-Sonnet-Thinking | 0.247 | Small |
| o3-mini | Llama-3.3-70B-Instruct | 0.247 | Small |
| o3-mini | Gemini-2.0-Flash | 0.247 | Small |
| GPT-4.5-Preview | Claude-3.7-Sonnet-Thinking | 0.247 | Small |
| GPT-4.5-Preview | Llama-3.3-70B-Instruct | 0.247 | Small |
| GPT-4.5-Preview | Gemini-2.0-Flash | 0.247 | Small |
| GPT-4o | Gemini-2.0-Flash-Thinking-Exp0121 | 0.169 | Small |
| Claude-3.7-Sonnet-Thinking | DeepSeek-R1 | 0.166 | Small |
| Llama-3.3-70B-Instruct | DeepSeek-R1 | 0.166 | Small |
| DeepSeek-R1 | Gemini-2.0-Flash | 0.166 | Small |
| o3-mini | o3-mini-high | 0.164 | Small |
| o3-mini | Claude-3.7-Sonnet | 0.164 | Small |
| GPT-4.5-Preview | o3-mini-high | 0.164 | Small |
| GPT-4.5-Preview | Claude-3.7-Sonnet | 0.164 | Small |
| o3-mini-high | GPT-4o | 0.164 | Small |
| GPT-4o | Claude-3.7-Sonnet | 0.164 | Small |
| Gemma-3 | GPT-4o-mini | 0.092 | Small |
| DeepSeek-R1 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.085 | Small |
| GPT-4o | DeepSeek-R1 | 0.083 | Small |
| GPT-4o | Claude-3.7-Sonnet-Thinking | 0.082 | Small |
| GPT-4o | Llama-3.3-70B-Instruct | 0.082 | Small |
| GPT-4o | Gemini-2.0-Flash | 0.082 | Small |
| o3-mini-high | Claude-3.7-Sonnet-Thinking | 0.082 | Small |
| o3-mini-high | Llama-3.3-70B-Instruct | 0.082 | Small |
| o3-mini-high | Gemini-2.0-Flash | 0.082 | Small |
| Claude-3.7-Sonnet | Claude-3.7-Sonnet-Thinking | 0.082 | Small |
| Claude-3.7-Sonnet | Llama-3.3-70B-Instruct | 0.082 | Small |
| Claude-3.7-Sonnet | Gemini-2.0-Flash | 0.082 | Small |
| o3-mini | GPT-4.5-Preview | 0.000 | Small |
| o3-mini-high | Claude-3.7-Sonnet | 0.000 | Small |
| Claude-3.7-Sonnet-Thinking | Llama-3.3-70B-Instruct | 0.000 | Small |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash | 0.000 | Small |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash | 0.000 | Small |

### Accuracy: Sexual Behavioral Analysis

*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*

#### Repeated Measures ANOVA

Repeated Measures ANOVA test for differences in accuracy: sexual behavioral analysis between models (accounting for test case variability):

- F-value: nan
- p-value: nan
- Degrees of freedom: 11, 55

The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a statistically significant difference in accuracy: sexual behavioral analysis between models when controlling for test case variability.

This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, which provides a more accurate assessment than treating each test case as independent.

#### Descriptive Statistics by Model

| Model | Mean | Std | Min | Max | Count | 95% CI |
|-------|------|-----|-----|-----|-------|---------|
| Claude-3.7-Sonnet | 0.500 | 0.516 | 0.000 | 1.000 | 16 | [0.225, 0.775] |
| Claude-3.7-Sonnet-Thinking | 0.500 | 0.516 | 0.000 | 1.000 | 16 | [0.225, 0.775] |
| DeepSeek-R1 | 0.438 | 0.512 | 0.000 | 1.000 | 16 | [0.164, 0.711] |
| GPT-4.5-Preview | 0.312 | 0.479 | 0.000 | 1.000 | 16 | [0.057, 0.568] |
| GPT-4o | 0.250 | 0.447 | 0.000 | 1.000 | 16 | [0.012, 0.488] |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.188 | 0.403 | 0.000 | 1.000 | 16 | [-0.027, 0.402] |
| GPT-4o-mini | 0.125 | 0.342 | 0.000 | 1.000 | 16 | [-0.057, 0.307] |
| Llama-3.3-70B-Instruct | 0.125 | 0.342 | 0.000 | 1.000 | 16 | [-0.057, 0.307] |
| Gemini-2.0-Flash | 0.125 | 0.342 | 0.000 | 1.000 | 16 | [-0.057, 0.307] |
| o3-mini | 0.062 | 0.250 | 0.000 | 1.000 | 16 | [-0.071, 0.196] |
| Gemma-3 | 0.000 | 0.000 | 0.000 | 0.000 | 16 | [nan, nan] |
| o3-mini-high | 0.000 | 0.000 | 0.000 | 0.000 | 16 | [nan, nan] |

#### Effect Size Analysis (Cohen's d)

Cohen's d measures the standardized difference between two means. It indicates the magnitude of the effect:

- Small effect: d < 0.5
- Medium effect: 0.5 ≤ d < 0.8
- Large effect: d ≥ 0.8

**Note:** Benjamini-Hochberg False Discovery Rate (FDR) correction has been applied to control for 66 multiple comparisons.
When multiple statistical tests are performed, the probability of observing a significant result by chance increases.
The Benjamini-Hochberg procedure adjusts p-values to control the expected proportion of false discoveries among all rejected hypotheses.
Effect sizes labeled with '(corrected)' indicate comparisons that were not statistically significant after correction.

| Model 1 | Model 2 | Cohen's d | Effect Size |
|---------|---------|-----------|-------------|
| Gemma-3 | Claude-3.7-Sonnet | 1.369 | Large |
| Gemma-3 | Claude-3.7-Sonnet-Thinking | 1.369 | Large |
| o3-mini-high | Claude-3.7-Sonnet | 1.369 | Large |
| o3-mini-high | Claude-3.7-Sonnet-Thinking | 1.369 | Large |
| Gemma-3 | DeepSeek-R1 | 1.208 | Large |
| o3-mini-high | DeepSeek-R1 | 1.208 | Large |
| o3-mini | Claude-3.7-Sonnet | 1.078 | Large |
| o3-mini | Claude-3.7-Sonnet-Thinking | 1.078 | Large |
| o3-mini | DeepSeek-R1 | 0.930 | Medium (corrected) |
| Gemma-3 | GPT-4.5-Preview | 0.923 | Medium (corrected) |
| GPT-4.5-Preview | o3-mini-high | 0.923 | Medium (corrected) |
| GPT-4o-mini | Claude-3.7-Sonnet | 0.857 | Medium (corrected) |
| GPT-4o-mini | Claude-3.7-Sonnet-Thinking | 0.857 | Medium (corrected) |
| Claude-3.7-Sonnet | Llama-3.3-70B-Instruct | 0.857 | Medium (corrected) |
| Claude-3.7-Sonnet | Gemini-2.0-Flash | 0.857 | Medium (corrected) |
| Claude-3.7-Sonnet-Thinking | Llama-3.3-70B-Instruct | 0.857 | Medium (corrected) |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash | 0.857 | Medium (corrected) |
| Gemma-3 | GPT-4o | 0.791 | Small (corrected) |
| o3-mini-high | GPT-4o | 0.791 | Small (corrected) |
| GPT-4o-mini | DeepSeek-R1 | 0.718 | Small (corrected) |
| Llama-3.3-70B-Instruct | DeepSeek-R1 | 0.718 | Small (corrected) |
| DeepSeek-R1 | Gemini-2.0-Flash | 0.718 | Small (corrected) |
| Claude-3.7-Sonnet | Gemini-2.0-Flash-Thinking-Exp0121 | 0.675 | Small (corrected) |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash-Thinking-Exp0121 | 0.675 | Small (corrected) |
| Gemma-3 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.658 | Small (corrected) |
| o3-mini-high | Gemini-2.0-Flash-Thinking-Exp0121 | 0.658 | Small (corrected) |
| o3-mini | GPT-4.5-Preview | 0.655 | Small (corrected) |
| DeepSeek-R1 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.542 | Small (corrected) |
| Gemma-3 | GPT-4o-mini | 0.518 | Small (corrected) |
| Gemma-3 | Llama-3.3-70B-Instruct | 0.518 | Small (corrected) |
| Gemma-3 | Gemini-2.0-Flash | 0.518 | Small (corrected) |
| o3-mini-high | GPT-4o-mini | 0.518 | Small (corrected) |
| o3-mini-high | Llama-3.3-70B-Instruct | 0.518 | Small (corrected) |
| o3-mini-high | Gemini-2.0-Flash | 0.518 | Small (corrected) |
| GPT-4o | Claude-3.7-Sonnet | 0.518 | Small (corrected) |
| GPT-4o | Claude-3.7-Sonnet-Thinking | 0.518 | Small (corrected) |
| o3-mini | GPT-4o | 0.518 | Small (corrected) |
| GPT-4.5-Preview | GPT-4o-mini | 0.451 | Small |
| GPT-4.5-Preview | Llama-3.3-70B-Instruct | 0.451 | Small |
| GPT-4.5-Preview | Gemini-2.0-Flash | 0.451 | Small |
| GPT-4o | DeepSeek-R1 | 0.390 | Small |
| GPT-4.5-Preview | Claude-3.7-Sonnet | 0.377 | Small |
| GPT-4.5-Preview | Claude-3.7-Sonnet-Thinking | 0.377 | Small |
| o3-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.373 | Small |
| o3-mini | Gemma-3 | 0.354 | Small |
| o3-mini | o3-mini-high | 0.354 | Small |
| GPT-4o | GPT-4o-mini | 0.314 | Small |
| GPT-4o | Llama-3.3-70B-Instruct | 0.314 | Small |
| GPT-4o | Gemini-2.0-Flash | 0.314 | Small |
| GPT-4.5-Preview | Gemini-2.0-Flash-Thinking-Exp0121 | 0.282 | Small |
| GPT-4.5-Preview | DeepSeek-R1 | 0.252 | Small |
| o3-mini | GPT-4o-mini | 0.209 | Small |
| o3-mini | Llama-3.3-70B-Instruct | 0.209 | Small |
| o3-mini | Gemini-2.0-Flash | 0.209 | Small |
| GPT-4o-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.167 | Small |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash-Thinking-Exp0121 | 0.167 | Small |
| Gemini-2.0-Flash | Gemini-2.0-Flash-Thinking-Exp0121 | 0.167 | Small |
| GPT-4o | Gemini-2.0-Flash-Thinking-Exp0121 | 0.147 | Small |
| GPT-4.5-Preview | GPT-4o | 0.135 | Small |
| Claude-3.7-Sonnet | DeepSeek-R1 | 0.122 | Small |
| Claude-3.7-Sonnet-Thinking | DeepSeek-R1 | 0.122 | Small |
| Gemma-3 | o3-mini-high | 0.000 | None (constant values) |
| GPT-4o-mini | Llama-3.3-70B-Instruct | 0.000 | Small |
| GPT-4o-mini | Gemini-2.0-Flash | 0.000 | Small |
| Claude-3.7-Sonnet | Claude-3.7-Sonnet-Thinking | 0.000 | Small |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash | 0.000 | Small |

### Accuracy: Sexual Homicide Pathways Analysis

*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*

#### Repeated Measures ANOVA

Repeated Measures ANOVA test for differences in accuracy: sexual homicide pathways analysis between models (accounting for test case variability):

- F-value: nan
- p-value: nan
- Degrees of freedom: 11, 55

The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a statistically significant difference in accuracy: sexual homicide pathways analysis between models when controlling for test case variability.

This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, which provides a more accurate assessment than treating each test case as independent.

#### Descriptive Statistics by Model

| Model | Mean | Std | Min | Max | Count | 95% CI |
|-------|------|-----|-----|-----|-------|---------|
| o3-mini | 0.750 | 0.447 | 0.000 | 1.000 | 16 | [0.512, 0.988] |
| o3-mini-high | 0.750 | 0.447 | 0.000 | 1.000 | 16 | [0.512, 0.988] |
| Claude-3.7-Sonnet | 0.750 | 0.447 | 0.000 | 1.000 | 16 | [0.512, 0.988] |
| Claude-3.7-Sonnet-Thinking | 0.750 | 0.447 | 0.000 | 1.000 | 16 | [0.512, 0.988] |
| Llama-3.3-70B-Instruct | 0.750 | 0.447 | 0.000 | 1.000 | 16 | [0.512, 0.988] |
| DeepSeek-R1 | 0.750 | 0.447 | 0.000 | 1.000 | 16 | [0.512, 0.988] |
| GPT-4o | 0.688 | 0.479 | 0.000 | 1.000 | 16 | [0.432, 0.943] |
| Gemma-3 | 0.625 | 0.500 | 0.000 | 1.000 | 16 | [0.359, 0.891] |
| GPT-4.5-Preview | 0.625 | 0.500 | 0.000 | 1.000 | 16 | [0.359, 0.891] |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.625 | 0.500 | 0.000 | 1.000 | 16 | [0.359, 0.891] |
| GPT-4o-mini | 0.562 | 0.512 | 0.000 | 1.000 | 16 | [0.289, 0.836] |
| Gemini-2.0-Flash | 0.438 | 0.512 | 0.000 | 1.000 | 16 | [0.164, 0.711] |

#### Effect Size Analysis (Cohen's d)

Cohen's d measures the standardized difference between two means. It indicates the magnitude of the effect:

- Small effect: d < 0.5
- Medium effect: 0.5 ≤ d < 0.8
- Large effect: d ≥ 0.8

**Note:** Benjamini-Hochberg False Discovery Rate (FDR) correction has been applied to control for 66 multiple comparisons.
When multiple statistical tests are performed, the probability of observing a significant result by chance increases.
The Benjamini-Hochberg procedure adjusts p-values to control the expected proportion of false discoveries among all rejected hypotheses.
Effect sizes labeled with '(corrected)' indicate comparisons that were not statistically significant after correction.

| Model 1 | Model 2 | Cohen's d | Effect Size |
|---------|---------|-----------|-------------|
| o3-mini | Gemini-2.0-Flash | 0.650 | Small (corrected) |
| o3-mini-high | Gemini-2.0-Flash | 0.650 | Small (corrected) |
| Claude-3.7-Sonnet | Gemini-2.0-Flash | 0.650 | Small (corrected) |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash | 0.650 | Small (corrected) |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash | 0.650 | Small (corrected) |
| DeepSeek-R1 | Gemini-2.0-Flash | 0.650 | Small (corrected) |
| GPT-4o | Gemini-2.0-Flash | 0.504 | Small (corrected) |
| o3-mini | GPT-4o-mini | 0.390 | Small |
| o3-mini-high | GPT-4o-mini | 0.390 | Small |
| GPT-4o-mini | Claude-3.7-Sonnet | 0.390 | Small |
| GPT-4o-mini | Claude-3.7-Sonnet-Thinking | 0.390 | Small |
| GPT-4o-mini | Llama-3.3-70B-Instruct | 0.390 | Small |
| GPT-4o-mini | DeepSeek-R1 | 0.390 | Small |
| Gemma-3 | Gemini-2.0-Flash | 0.370 | Small |
| GPT-4.5-Preview | Gemini-2.0-Flash | 0.370 | Small |
| Gemini-2.0-Flash | Gemini-2.0-Flash-Thinking-Exp0121 | 0.370 | Small |
| o3-mini | Gemma-3 | 0.264 | Small |
| o3-mini | GPT-4.5-Preview | 0.264 | Small |
| o3-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.264 | Small |
| Gemma-3 | o3-mini-high | 0.264 | Small |
| Gemma-3 | Claude-3.7-Sonnet | 0.264 | Small |
| Gemma-3 | Claude-3.7-Sonnet-Thinking | 0.264 | Small |
| Gemma-3 | Llama-3.3-70B-Instruct | 0.264 | Small |
| Gemma-3 | DeepSeek-R1 | 0.264 | Small |
| GPT-4.5-Preview | o3-mini-high | 0.264 | Small |
| GPT-4.5-Preview | Claude-3.7-Sonnet | 0.264 | Small |
| GPT-4.5-Preview | Claude-3.7-Sonnet-Thinking | 0.264 | Small |
| GPT-4.5-Preview | Llama-3.3-70B-Instruct | 0.264 | Small |
| GPT-4.5-Preview | DeepSeek-R1 | 0.264 | Small |
| o3-mini-high | Gemini-2.0-Flash-Thinking-Exp0121 | 0.264 | Small |
| Claude-3.7-Sonnet | Gemini-2.0-Flash-Thinking-Exp0121 | 0.264 | Small |
| Claude-3.7-Sonnet-Thinking | Gemini-2.0-Flash-Thinking-Exp0121 | 0.264 | Small |
| Llama-3.3-70B-Instruct | Gemini-2.0-Flash-Thinking-Exp0121 | 0.264 | Small |
| DeepSeek-R1 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.264 | Small |
| GPT-4o | GPT-4o-mini | 0.252 | Small |
| GPT-4o-mini | Gemini-2.0-Flash | 0.244 | Small |
| o3-mini | GPT-4o | 0.135 | Small |
| o3-mini-high | GPT-4o | 0.135 | Small |
| GPT-4o | Claude-3.7-Sonnet | 0.135 | Small |
| GPT-4o | Claude-3.7-Sonnet-Thinking | 0.135 | Small |
| GPT-4o | Llama-3.3-70B-Instruct | 0.135 | Small |
| GPT-4o | DeepSeek-R1 | 0.135 | Small |
| Gemma-3 | GPT-4o | 0.128 | Small |
| GPT-4.5-Preview | GPT-4o | 0.128 | Small |
| GPT-4o | Gemini-2.0-Flash-Thinking-Exp0121 | 0.128 | Small |
| Gemma-3 | GPT-4o-mini | 0.123 | Small |
| GPT-4.5-Preview | GPT-4o-mini | 0.123 | Small |
| GPT-4o-mini | Gemini-2.0-Flash-Thinking-Exp0121 | 0.123 | Small |
| o3-mini | o3-mini-high | 0.000 | Small |
| o3-mini | Claude-3.7-Sonnet | 0.000 | Small |
| o3-mini | Claude-3.7-Sonnet-Thinking | 0.000 | Small |
| o3-mini | Llama-3.3-70B-Instruct | 0.000 | Small |
| o3-mini | DeepSeek-R1 | 0.000 | Small |
| Gemma-3 | GPT-4.5-Preview | 0.000 | Small |
| Gemma-3 | Gemini-2.0-Flash-Thinking-Exp0121 | 0.000 | Small |
| GPT-4.5-Preview | Gemini-2.0-Flash-Thinking-Exp0121 | 0.000 | Small |
| o3-mini-high | Claude-3.7-Sonnet | 0.000 | Small |
| o3-mini-high | Claude-3.7-Sonnet-Thinking | 0.000 | Small |
| o3-mini-high | Llama-3.3-70B-Instruct | 0.000 | Small |
| o3-mini-high | DeepSeek-R1 | 0.000 | Small |
| Claude-3.7-Sonnet | Claude-3.7-Sonnet-Thinking | 0.000 | Small |
| Claude-3.7-Sonnet | Llama-3.3-70B-Instruct | 0.000 | Small |
| Claude-3.7-Sonnet | DeepSeek-R1 | 0.000 | Small |
| Claude-3.7-Sonnet-Thinking | Llama-3.3-70B-Instruct | 0.000 | Small |
| Claude-3.7-Sonnet-Thinking | DeepSeek-R1 | 0.000 | Small |
| Llama-3.3-70B-Instruct | DeepSeek-R1 | 0.000 | Small |

## Per-Case Analysis

This section analyzes performance across different test cases to identify which cases are particularly challenging or easy for language models.

The evaluation dataset contains 6 distinct test cases.

### Overall Case Difficulty

This table shows the average performance across all models for each test case, sorted from highest to lowest performance.

| Test Case | Acc: Narrative Action System | Acc: Spatial Behavioral Analysis | Acc: Sexual Behavioral Analysis | Acc: Sexual Homicide Pathways Analysis | Average |
|-----------|----------|----------|----------|----------|----------|
| mad-bomber | 1.000 | 0.583 | nan | nan | 0.792 |
| unabomber | 0.562 | 1.000 | nan | nan | 0.781 |
| ted-bundy-lake | 0.729 | 0.729 | 0.375 | 0.792 | 0.656 |
| btk-otero | 0.688 | 0.479 | 0.458 | 0.938 | 0.641 |
| robert-napper-rachel-nickell | 0.312 | 0.104 | 0.021 | 0.958 | 0.349 |
| ed-kemper | 0.208 | 0.167 | 0.021 | 0.000 | 0.099 |

#### Easiest Test Cases

The following test cases had the highest average performance across all models and metrics:

1. **mad-bomber** - Average Score: 0.792
2. **unabomber** - Average Score: 0.781
3. **ted-bundy-lake** - Average Score: 0.656

#### Most Challenging Test Cases

The following test cases had the lowest average performance across all models and metrics:

1. **btk-otero** - Average Score: 0.641
2. **robert-napper-rachel-nickell** - Average Score: 0.349
3. **ed-kemper** - Average Score: 0.099

### Narrative Action System Framework Case Analysis

| Test Case | Gold Standard | Success Rate | Correct Models | Total Models |
|-----------|---------------|-------------|----------------|-------------|
| mad-bomber | REVENGEFUL | 1.000 | 12 | 12 |
| ted-bundy-lake | PROFESSIONAL | 0.729 | 11 | 12 |
| btk-otero | PROFESSIONAL | 0.688 | 11 | 12 |
| unabomber | REVENGEFUL | 0.562 | 11 | 12 |
| robert-napper-rachel-nickell | TRAGIC_HERO | 0.312 | 7 | 12 |
| ed-kemper | REVENGEFUL | 0.208 | 5 | 12 |

### Spatial Behavioral Analysis Framework Case Analysis

| Test Case | Gold Standard | Success Rate | Correct Models | Total Models |
|-----------|---------------|-------------|----------------|-------------|
| unabomber | COMMUTER | 1.000 | 12 | 12 |
| ted-bundy-lake | COMMUTER | 0.729 | 12 | 12 |
| mad-bomber | COMMUTER | 0.583 | 9 | 12 |
| btk-otero | MARAUDER | 0.479 | 9 | 12 |
| ed-kemper | MARAUDER | 0.167 | 3 | 12 |
| robert-napper-rachel-nickell | COMMUTER | 0.104 | 5 | 12 |

### Sexual Behavioral Analysis Framework Case Analysis

| Test Case | Gold Standard | Success Rate | Correct Models | Total Models |
|-----------|---------------|-------------|----------------|-------------|
| btk-otero | ANGER_EXCITATION | 0.458 | 8 | 12 |
| ted-bundy-lake | ANGER_EXCITATION | 0.375 | 9 | 12 |
| ed-kemper | ANGER_RETALIATORY | 0.021 | 1 | 12 |
| robert-napper-rachel-nickell | ANGER_EXCITATION | 0.021 | 1 | 12 |

### Sexual Homicide Pathways Analysis Framework Case Analysis

| Test Case | Gold Standard | Success Rate | Correct Models | Total Models |
|-----------|---------------|-------------|----------------|-------------|
| robert-napper-rachel-nickell | ANGRY | 0.958 | 12 | 12 |
| btk-otero | SADISTIC | 0.938 | 12 | 12 |
| ted-bundy-lake | SADISTIC | 0.792 | 12 | 12 |
| ed-kemper | ANGRY | 0.000 | 0 | 12 |

### Statistical Comparison of Case Difficulty

We analyzed whether there are statistically significant differences in Accuracy: Narrative Action System across test cases.

*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*

#### Repeated Measures ANOVA

Repeated Measures ANOVA test for differences in accuracy: narrative action system between models (accounting for test case variability):

- F-value: 1.3348
- p-value: 0.2308
- Degrees of freedom: 11, 55

The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a statistically significant difference in accuracy: narrative action system between models when controlling for test case variability.

This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, which provides a more accurate assessment than treating each test case as independent.

### Model Consistency Across Test Cases

This analysis examines whether models perform consistently across different test cases or if performance varies significantly by case.

#### Model Consistency in Accuracy: Narrative Action System

The coefficient of variation (CV) measures how consistent a model's performance is across different test cases.
Lower CV values indicate more consistent performance across test cases.

| Model | Mean | Std Dev | Coefficient of Variation | # Cases |
|-------|------|---------|--------------------------|--------|
| Llama-3.3-70B-Instruct | 0.708 | 0.172 | 0.243 | 6 |
| Gemini-2.0-Flash | 0.750 | 0.204 | 0.272 | 6 |
| DeepSeek-R1 | 0.625 | 0.346 | 0.554 | 6 |
| o3-mini-high | 0.667 | 0.373 | 0.559 | 6 |
| Gemma-3 | 0.458 | 0.267 | 0.582 | 6 |
| o3-mini | 0.708 | 0.419 | 0.591 | 6 |
| Claude-3.7-Sonnet | 0.667 | 0.471 | 0.707 | 6 |
| GPT-4o-mini | 0.500 | 0.354 | 0.707 | 6 |
| Claude-3.7-Sonnet-Thinking | 0.625 | 0.451 | 0.721 | 6 |
| GPT-4o | 0.583 | 0.449 | 0.769 | 6 |
| GPT-4.5-Preview | 0.375 | 0.375 | 1.000 | 6 |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | 0.373 | 1.118 | 6 |

**Llama-3.3-70B-Instruct** shows the most consistent performance across different test cases (CV: 0.243), while **Gemini-2.0-Flash-Thinking-Exp0121** shows the most variable performance (CV: 1.118).

## Reliability Analysis

To ensure scientific rigor, each case was tested multiple times with each model. This section analyzes the consistency of model performance across repeated runs of the same test cases.

Each case-model combination was tested up to 4 times to assess reliability.

### Within-Model Reliability

This table shows consistency metrics for each model across repeated runs of the same test cases.

#### Reliability for Accuracy: Narrative Action System

| Model | CV (lower is better) | Reliability Rating | % Identical Results |
|----|----|----|--|
| Claude-3.7-Sonnet-Thinking | 0.144 | Good | 83.3% |
| Claude-3.7-Sonnet | 0.000 | Excellent | 100.0% |
| o3-mini-high | 0.400 | Poor | 66.7% |
| DeepSeek-R1 | 0.515 | Poor | 50.0% |
| o3-mini | 0.346 | Poor | 83.3% |
| GPT-4o | 0.250 | Moderate | 83.3% |
| GPT-4.5-Preview | 1.010 | Poor | 50.0% |
| Gemma-3 | 1.199 | Poor | 16.7% |
| Llama-3.3-70B-Instruct | 0.622 | Poor | 16.7% |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.667 | Poor | 66.7% |
| Gemini-2.0-Flash | 0.526 | Poor | 33.3% |
| GPT-4o-mini | 0.924 | Poor | 33.3% |

**Claude-3.7-Sonnet-Thinking** shows the most consistent results across repeated runs (CV: 0.144), while **GPT-4o-mini** shows the most variable results (CV: 0.924).

**Claude-3.7-Sonnet** achieved perfect consistency, producing identical results across all repeated runs.

#### Reliability for Accuracy: Spatial Behavioral Analysis

| Model | CV (lower is better) | Reliability Rating | % Identical Results |
|----|----|----|--|
| Claude-3.7-Sonnet-Thinking | 0.192 | Good | 83.3% |
| Claude-3.7-Sonnet | 0.577 | Poor | 66.7% |
| o3-mini-high | 0.577 | Poor | 66.7% |
| DeepSeek-R1 | 0.933 | Poor | 50.0% |
| o3-mini | 0.250 | Moderate | 83.3% |
| GPT-4o | 0.866 | Poor | 66.7% |
| GPT-4.5-Preview | 0.662 | Poor | 50.0% |
| Gemma-3 | 0.648 | Poor | 33.3% |
| Llama-3.3-70B-Instruct | 0.722 | Poor | 50.0% |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.770 | Poor | 66.7% |
| Gemini-2.0-Flash | 0.192 | Good | 83.3% |
| GPT-4o-mini | 0.577 | Poor | 66.7% |

**Claude-3.7-Sonnet-Thinking** shows the most consistent results across repeated runs (CV: 0.192), while **GPT-4o-mini** shows the most variable results (CV: 0.577).

#### Reliability for Accuracy: Sexual Behavioral Analysis

| Model | CV (lower is better) | Reliability Rating | % Identical Results |
|----|----|----|--|
| Claude-3.7-Sonnet-Thinking | 0.000 | Excellent | 100.0% |
| Claude-3.7-Sonnet | 0.000 | Excellent | 100.0% |
| o3-mini-high | nan | nan | nan% |
| DeepSeek-R1 | 0.289 | Moderate | 75.0% |
| o3-mini | 1.732 | Poor | 75.0% |
| GPT-4o | 1.155 | Poor | 50.0% |
| GPT-4.5-Preview | 0.866 | Poor | 75.0% |
| Gemma-3 | nan | nan | nan% |
| Llama-3.3-70B-Instruct | 1.732 | Poor | 50.0% |
| Gemini-2.0-Flash-Thinking-Exp0121 | 1.366 | Poor | 50.0% |
| Gemini-2.0-Flash | 1.732 | Poor | 50.0% |
| GPT-4o-mini | 1.732 | Poor | 50.0% |

**Claude-3.7-Sonnet-Thinking** shows the most consistent results across repeated runs (CV: 0.000), while **GPT-4o-mini** shows the most variable results (CV: 1.732).

The following models achieved perfect consistency, producing identical results across all repeated runs: **Claude-3.7-Sonnet-Thinking**, **Claude-3.7-Sonnet**.

#### Reliability for Accuracy: Sexual Homicide Pathways Analysis

| Model | CV (lower is better) | Reliability Rating | % Identical Results |
|----|----|----|--|
| Claude-3.7-Sonnet-Thinking | 0.000 | Excellent | 100.0% |
| Claude-3.7-Sonnet | 0.000 | Excellent | 100.0% |
| o3-mini-high | 0.000 | Excellent | 100.0% |
| DeepSeek-R1 | 0.000 | Excellent | 100.0% |
| o3-mini | 0.000 | Excellent | 100.0% |
| GPT-4o | 0.192 | Good | 75.0% |
| GPT-4.5-Preview | 0.333 | Poor | 75.0% |
| Gemma-3 | 0.385 | Poor | 50.0% |
| Llama-3.3-70B-Instruct | 0.000 | Excellent | 100.0% |
| Gemini-2.0-Flash-Thinking-Exp0121 | 0.333 | Poor | 75.0% |
| Gemini-2.0-Flash | 0.911 | Poor | 50.0% |
| GPT-4o-mini | 0.526 | Poor | 50.0% |

**Claude-3.7-Sonnet-Thinking** shows the most consistent results across repeated runs (CV: 0.000), while **GPT-4o-mini** shows the most variable results (CV: 0.526).

The following models achieved perfect consistency, producing identical results across all repeated runs: **Claude-3.7-Sonnet-Thinking**, **Claude-3.7-Sonnet**, **o3-mini-high**, **DeepSeek-R1**, **o3-mini**, **Llama-3.3-70B-Instruct**.

### Interpretation of Reliability Metrics

The Coefficient of Variation (CV) measures the consistency of results across repeated runs of the same test cases:

- **Excellent reliability**: CV < 0.1 (less than 10% variation)
- **Good reliability**: CV < 0.2 (less than 20% variation)
- **Moderate reliability**: CV < 0.3 (less than 30% variation)
- **Poor reliability**: CV ≥ 0.3 (30% or greater variation)

The **% Identical Results** shows how often a model produced exactly the same result on all runs of the same test case.

### Scientific Implications

Reliability analysis is critical for scientific rigor when evaluating language models. High reliability indicates:

1. **Deterministic behavior**: Models producing identical results on repeated runs demonstrate deterministic outputs.
2. **Performance stability**: Low variation across runs suggests reliable performance metrics that can be trusted.
3. **Scientific validity**: Models with higher reliability produce results that are more scientifically valid and reproducible.

When interpreting performance metrics in this report, consider each model's reliability alongside its raw performance. Models with inconsistent results (high CV) may show inflated performance in some metrics due to chance rather than true capability.

### Repeated Measures Analysis

Since each case was tested multiple times with each model, we conducted a Repeated Measures ANOVA to account for this experimental design.

We analyzed Accuracy: Sexual Homicide Pathways Analysis using our Repeated Measures ANOVA approach to properly account for the correlation structure in repeated measurements.

See the 'Statistical Methodology' section for details about how Repeated Measures ANOVA provides a more rigorous analysis than traditional ANOVA for this experimental design.

#### Statistical Methodology

**Note on statistical methodology**: This analysis employs Repeated Measures ANOVA, which directly models the within-subject factor (models) and the between-subject factor (test cases). Unlike traditional one-way ANOVA, this approach properly accounts for the correlation structure in repeated measurements, providing greater statistical power and more accurate p-values by controlling for the natural variability between test cases.

This method avoids pseudoreplication and inflated degrees of freedom that would occur if treating each repetition as an independent observation, resulting in a more rigorous and scientifically sound analysis than traditional ANOVA approaches.

We analyzed Accuracy: Narrative Action System using an approach that accounts for repeated measurements.

#### Repeated Measures ANOVA

Repeated Measures ANOVA test for differences in accuracy: narrative action system between models (accounting for test case variability):

- F-value: 1.3348
- p-value: 0.2308
- Degrees of freedom: 11, 55

The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a statistically significant difference in accuracy: narrative action system between models when controlling for test case variability.

This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, which provides a more accurate assessment than treating each test case as independent.

## Conclusion

Based on the comprehensive analysis of all metrics, **Claude-3.7-Sonnet** demonstrates the strongest overall performance with an average score of 0.647 across all evaluation criteria.

This report provides a comprehensive overview of the performance of various language models on forensic analysis tasks. The data can be used to draw conclusions about the relative strengths and weaknesses of different models in understanding and applying forensic frameworks.

*End of Report*
