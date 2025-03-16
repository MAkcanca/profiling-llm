# Forensic-LLM2: Framework for Criminal Profiling with Large Language Models

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Forensic-LLM2 is a groundbreaking research framework that evaluates the efficacy of large language models (LLMs) for forensic criminal profiling tasks. This research implements a structured methodology for extracting offender profiles from criminal case data using various state-of-the-art LLMs, evaluating their performance against gold standards, and providing comprehensive analysis of the results.

### Key Contributions

- Implementation of a structured criminal profiling framework based on established psychological frameworks
- Evaluation of 11+ state-of-the-art LLM systems on real-world criminal profiling tasks
- Development of quantitative metrics for assessing profile quality and accuracy
- Comparative analysis of model performance across different profiling dimensions
- Identification of key factors influencing profiling accuracy
- Open-source methodology to encourage further research in computational forensic psychology

## Methodology

### Profiling Frameworks

The project applies four established criminological frameworks to structure profile generation and evaluation:

#### 1. Narrative Action System (NAS)

The NAS framework categorizes offenders based on their narrative roles and behavioral patterns during criminal activities:

- **Revengeful Mission**: Offenders driven by a perceived justification for their actions, targeting victims with specific significance
- **Tragic Hero**: Individuals who view themselves as powerless protagonists acting out an inevitable fate
- **Professional**: Methodical offenders focused on efficiency and demonstrating technical prowess
- **Victim**: Confused protagonists acting from perceived powerlessness, often exhibiting disorganized behaviors

#### 2. Sexual Behavioral Analysis (SBA)

This framework specifically examines sexual offense patterns:

- **Power-Assertive**: Offenders demonstrating dominance through control, where violence serves as an extension of masculine identity
- **Power-Reassurance**: Motivated by fantasy and seeking validation, characterized by minimal force until fantasy disruption
- **Anger-Retaliatory**: Driven by anger and revenge, characterized by excessive violence and blitz attacks
- **Anger-Excitation**: Sadistic offenders who derive pleasure from victim suffering, employing extended control and torture

#### 3. Sexual Homicide Pathways Analysis (SHPA)

A developmental pathway approach to sexual homicide:

- **Sadistic Pathway**: Characterized by elaborate planning, ritual behaviors, and sexual gratification through victim suffering
- **Angry Pathway**: Driven by rage and characterized by excessive violence beyond what's needed for homicide
- **Opportunistic Pathway**: Sexual homicide occurring as an extension of sexual assault, where homicide serves an instrumental purpose

#### 4. Spatial Behavioral Analysis

Analysis of geographic behavior patterns in criminal activities:

- **Marauder Pattern**: Offenders operating from a fixed anchor point within their comfort zone
- **Commuter Pattern**: Offenders who travel outside their home/anchor area to commit crimes

### Research Design

Our research implemented a multi-stage process:

1. **Case Selection**: Six real-world criminal cases were selected, representing diverse offender typologies and behavioral patterns
2. **Gold Standard Development**: Expert forensic psychologists created comprehensive gold standard profiles for each case, following the structured framework
3. **Model Selection**: 11+ state-of-the-art LLM systems were selected, representing different architectures, training approaches, and capabilities
4. **Profile Generation**: Each model generated structured profiles for all test cases using a consistent prompt methodology
5. **Multi-dimensional Evaluation**: Generated profiles were assessed through:
   - Framework Classification Accuracy
   - Reasoning Quality Assessment
   - Gold Standard Similarity Analysis
   - Error Pattern Identification

### Evaluation Metrics

The evaluation framework comprises multiple complementary metrics:

#### Framework Classification Analysis

Measures the models' ability to correctly classify offenders according to established theoretical frameworks:
- Classification accuracy per framework (NAS, SBA, SHPA, Spatial)
- Confidence calibration assessment
- Supporting evidence quality

#### Reasoning Quality Assessment

Evaluates the coherence and evidence basis of models' reasoning:
- Logical consistency score
- Evidence utilization index
- Alternative hypothesis consideration
- Speculative vs. evidence-based reasoning ratio

#### Gold Standard Similarity

Compares generated profiles to expert gold standards across multiple dimensions:
- Semantic similarity using embedding-based comparison
- Structural alignment (correctly identifying key profile elements)
- Factual consistency with case evidence
- Key insight identification rate

#### Error Analysis

Identifies patterns in model failures and limitations:
- Hallucination frequency and type
- Framework misapplication patterns
- Cognitive bias indicators
- Domain-specific error categories

## Models Evaluated

Our research evaluated 11+ state-of-the-art LLM systems, including:

| Model | Provider | Key Characteristics |
|-------|----------|---------------------|
| GPT-4.5-Preview | OpenAI | Latest GPT architecture with enhanced reasoning capabilities |
| GPT-4o | OpenAI | Optimized multimodal model |
| GPT-4o-mini | OpenAI | Compact version with reduced parameter count |
| Claude-3.7-Sonnet | Anthropic | Optimized for detail-oriented analytical tasks |
| Claude-3.7-Sonnet-Thinking | Anthropic | Enhanced with explicit reasoning capabilities |
| Llama-3.3-70B-Instruct | Meta | Open weights model with 70B parameters |
| DeepSeek-R1 | DeepSeek | Research-focused model with enhanced reasoning |
| Gemini-2.0-Flash | Google | Optimized for efficiency with strong reasoning |
| o3-mini | OpenAI | Compact but efficient model |
| o3-mini-high | OpenAI | With enhanced reasoning effort configuration |

Models were evaluated with their default settings as well as with enhanced reasoning configurations when available.

## Key Findings

### Performance Patterns

1. **Framework Accuracy**: Models demonstrated varying capabilities in correctly applying criminological frameworks, with specialized models showing stronger performance in structured classification tasks
2. **Reasoning Quality**: Advanced models with explicit reasoning capabilities showed measurable improvements in evidence-based reasoning and logical consistency
3. **Case Complexity Impact**: Performance degraded predictably with case complexity, particularly with ambiguous evidence or complex victimology
4. **Domain Knowledge Gaps**: Specific criminological concepts were consistently misapplied across multiple models, suggesting domain knowledge limitations

### Comparative Model Performance

1. Larger, more recent models generally outperformed smaller or earlier models, with explicit reasoning capabilities providing significant advantages
2. Performance deltas between models were most pronounced in:
   - Complex case reasoning
   - Alternative hypothesis consideration
   - Disciplined evidence utilization
3. Some specialized models showed surprising strength in specific criminological frameworks despite general performance limitations

### Methodological Implications

1. **Prompt Engineering Effects**: Structured prompting with explicit framework guidance significantly improved performance across all models
2. **Reasoning Pathway Impact**: Models with explicit reasoning pathways (thinking-aloud approaches) showed reduced framework misapplication
3. **Gold Standard Deviation Patterns**: Systematic deviations from expert profiles revealed consistent model biases and knowledge gaps

## Research Implications

### Methodological Contributions

1. **Framework Integration**: Our structured evaluation approach provides a template for rigorous assessment of LLMs in specialized domains
2. **Multi-dimensional Evaluation**: The complementary metrics provide a comprehensive understanding of model capabilities and limitations
3. **Gold Standard Methodology**: Our approach to expert-based validation establishes a replicable process for similar research

### Ethical Considerations

1. **Application Boundaries**: Clear limitations identified that preclude autonomous use in actual criminal investigations
2. **Bias Amplification Risks**: Models demonstrated potential to amplify existing biases in criminological literature
3. **Transparency Requirements**: The research highlights the need for explicit reasoning pathways when models are applied in sensitive domains

## Reproduction Instructions

To reproduce the main results from our paper:

1. Generate profiles for all test cases using the profile generation script
2. Run evaluation against gold standards using the evaluation framework
3. Generate statistical analyses and visualizations using the plotting utilities

The full reproduction package includes:
- Test cases in standardized format
- Gold standard profiles for validation
- Evaluation metrics implementation
- Statistical analysis scripts

## Citation Information

If you use Forensic-LLM2 in your research, please cite our paper:

```
@article{profilingllm,
  title={Evaluating Large Language Models for Criminal Profiling: A Framework-Based Approach},
  author={[Mustafa Akcanca]},
  journal={[ACDSA]},
  year={2025}
} 
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
