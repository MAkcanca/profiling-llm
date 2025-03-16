"""Export utilities for evaluation results."""
import json
import pandas as pd
import logging
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the statistical analysis functions
try:
    from ..analysis.statistical import perform_anova, calculate_effect_size, calculate_confidence_intervals
    from ..analysis.reliability import calculate_inter_model_agreement
    STATISTICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    STATISTICAL_ANALYSIS_AVAILABLE = False
    logging.warning("Statistical analysis module not available. Statistical metrics will be skipped.")

# Set up logger
logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable values."""
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, Path):
            return str(obj)
        elif obj is None:
            return None
        else:
            try:
                # Try default serialization
                return super().default(obj)
            except:
                # Return as string if all else fails
                return str(obj)

def save_results_to_json(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Path to save the JSON file
    """
    try:
        # Clean results before saving
        clean_results = []
        for result in results:
            # Create a clean copy with serializable values
            clean_result = {}
            for key, value in result.items():
                # Handle nested dictionaries
                if isinstance(value, dict):
                    clean_result[key] = {}
                    for k, v in value.items():
                        if isinstance(v, bool):
                            clean_result[key][k] = bool(v)
                        elif isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                            clean_result[key][k] = str(v)
                        else:
                            clean_result[key][k] = v
                # Handle direct values
                elif isinstance(value, bool):
                    clean_result[key] = bool(value)
                elif isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
                    clean_result[key] = str(value)
                else:
                    clean_result[key] = value
            clean_results.append(clean_result)
            
        with open(output_path, "w") as f:
            json.dump(clean_results, f, indent=2, cls=JSONEncoder)
        logger.info(f"Results saved to JSON: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")

def save_results_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save evaluation results to a CSV file.
    
    Args:
        df: DataFrame with evaluation results
        output_path: Path to save the CSV file
    """
    try:
        # Replace NaN and inf values with None for CSV
        df = df.replace([float('inf'), -float('inf')], None)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to CSV: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")

def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary DataFrame from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        DataFrame with summarized evaluation metrics
    """
    try:
        flattened_results = []
        for result in results:
            # Skip invalid results
            if not isinstance(result, dict):
                continue
                
            flat_result = {
                "test_case": str(result.get("test_case", "")),
                "model": str(result.get("model", "")),
                "version": str(result.get("version", "latest")),
                "timestamp": result.get("timestamp", 0)
            }
            
            # Add processing time if available and numeric
            proc_time = result.get("processing_time")
            if proc_time is not None and isinstance(proc_time, (int, float)):
                flat_result["processing_time"] = proc_time
            
            # Add reasoning metrics
            reasoning = result.get("reasoning")
            if reasoning and isinstance(reasoning, dict):
                # Add total count if available and numeric
                total_count = reasoning.get("total_count")
                if total_count is not None and isinstance(total_count, (int, float)):
                    flat_result["reasoning_count"] = total_count
                    
                # Add section counts if available
                section_counts = reasoning.get("section_counts")
                if section_counts and isinstance(section_counts, dict):
                    for section, count in section_counts.items():
                        if isinstance(count, (int, float)):
                            flat_result[f"reasoning_{section}"] = count
            
            # Add framework metrics
            framework = result.get("framework")
            if framework and isinstance(framework, dict):
                # Add average confidence if available and numeric
                avg_conf = framework.get("avg_confidence")
                if avg_conf is not None and isinstance(avg_conf, (int, float)):
                    flat_result["avg_framework_confidence"] = avg_conf
                
                # Add individual framework confidences
                framework_confidence = framework.get("framework_confidence")
                if framework_confidence and isinstance(framework_confidence, dict):
                    for fw_name, confidence in framework_confidence.items():
                        if isinstance(confidence, (int, float)):
                            flat_result[f"framework_confidence_{fw_name}"] = confidence
                
                # Add framework completeness
                framework_completeness = framework.get("framework_completeness")
                if framework_completeness and isinstance(framework_completeness, dict):
                    for fw_name, completeness in framework_completeness.items():
                        if isinstance(completeness, (int, float)):
                            flat_result[f"framework_completeness_{fw_name}"] = completeness
                            
                # Add primary framework classifications
                framework_primary = framework.get("framework_primary")
                if framework_primary and isinstance(framework_primary, dict):
                    for fw_name, classification in framework_primary.items():
                        if classification:
                            flat_result[f"framework_primary_{fw_name}"] = str(classification)
            
            # Add gold standard metrics
            gold_standard = result.get("gold_standard")
            if gold_standard and isinstance(gold_standard, dict):
                # Add framework agreement if available and numeric
                framework_agreement = gold_standard.get("framework_agreement")
                if framework_agreement is not None and isinstance(framework_agreement, (int, float)):
                    flat_result["framework_agreement"] = framework_agreement
                    
                # Add semantic similarity if available and numeric
                sem_sim = gold_standard.get("semantic_similarity")
                if sem_sim is not None and isinstance(sem_sim, (int, float)):
                    flat_result["semantic_similarity"] = sem_sim
                
                # Add accuracy metrics if available
                accuracy = gold_standard.get("accuracy")
                if accuracy and isinstance(accuracy, dict):
                    for fw_name, acc_value in accuracy.items():
                        if isinstance(acc_value, (int, float)):
                            flat_result[f"framework_accuracy_{fw_name}"] = acc_value
                
                # Extract framework matches/mismatches from gold standard comparisons
                framework_matches = gold_standard.get("framework_matches")
                if framework_matches and isinstance(framework_matches, dict):
                    for fw_name, match in framework_matches.items():
                        # Convert boolean or string match status to binary accuracy (1.0 = match, 0.0 = mismatch)
                        if isinstance(match, bool):
                            flat_result[f"framework_accuracy_{fw_name}"] = 1.0 if match else 0.0
                        elif isinstance(match, dict) and "match" in match:
                            flat_result[f"framework_accuracy_{fw_name}"] = 1.0 if match["match"] else 0.0
            
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    except Exception as e:
        logger.error(f"Error creating summary DataFrame: {e}")
        return pd.DataFrame()

def perform_statistical_analysis(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Perform statistical analysis on evaluation results.
    
    Args:
        df: DataFrame with evaluation results
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with statistical analysis results
    """
    if not STATISTICAL_ANALYSIS_AVAILABLE:
        return {"error": "Statistical analysis module not available"}
        
    try:
        analysis_results = {}
        # Define essential metrics to always analyze
        essential_metrics = [
            "reasoning_count", 
            "avg_framework_confidence", 
            "framework_agreement", 
            "semantic_similarity"
        ]
        
        # Create analysis directory
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Find framework accuracy metrics
        framework_accuracy_cols = [col for col in df.columns if col.startswith("framework_accuracy_")]
        
        # Combine essential metrics with framework accuracy metrics
        all_metrics = essential_metrics + framework_accuracy_cols
        
        # Only analyze metrics that exist and have sufficient data
        valid_metrics = []
        for metric in all_metrics:
            if metric in df.columns and not df[metric].isna().all():
                # Need at least 2 values per group for statistical analysis
                groups = df.groupby('model')[metric].count()
                if (groups >= 2).any():
                    valid_metrics.append(metric)
        
        # Skip analysis if no valid metrics
        if not valid_metrics:
            return {"error": "No valid metrics for statistical analysis"}
        
        # Generate statistical analyses for each metric
        for metric in valid_metrics:
            metric_results = {}
            
            # ANOVA
            anova_results = perform_anova(df, metric)
            metric_results["anova"] = anova_results
            
            # Effect Size
            effect_size_results = calculate_effect_size(df, metric)
            metric_results["effect_size"] = effect_size_results
            
            # Confidence Intervals
            ci_results = calculate_confidence_intervals(df, metric)
            metric_results["confidence_intervals"] = ci_results
            
            analysis_results[metric] = metric_results
        
        # If we have categorical data, calculate inter-model agreement
        framework_cols = [col for col in df.columns if col.startswith("framework_confidence_")]
        for framework_col in framework_cols:
            framework_name = framework_col.replace("framework_confidence_", "")
            primary_col = f"framework_primary_{framework_name}"
            
            if primary_col in df.columns and len(df[primary_col].dropna()) >= 4:
                agreement_results = calculate_inter_model_agreement(df, primary_col)
                analysis_results[f"agreement_{framework_name}"] = agreement_results
        
        # Save analysis results to JSON
        with open(analysis_dir / "statistical_analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2, cls=JSONEncoder)
            
        # Create a simplified summary for the paper
        paper_summary = create_paper_summary(analysis_results)
        with open(analysis_dir / "paper_summary.txt", "w") as f:
            f.write(paper_summary)
            
        return analysis_results
    except Exception as e:
        logger.error(f"Error performing statistical analysis: {e}")
        return {"error": str(e)}

def create_paper_summary(analysis_results: Dict[str, Any]) -> str:
    """Create a summary of statistical results suitable for inclusion in a paper.
    
    Args:
        analysis_results: Dictionary with statistical analysis results
        
    Returns:
        String with formatted summary for paper
    """
    lines = ["# Statistical Analysis Summary for Scientific Paper", ""]
    lines.append("## Key Findings Summary")
    lines.append("This section provides a concise summary of the most important statistical findings.")
    lines.append("")
    
    # Track essential metrics to provide a summary at the top
    key_findings = []
    
    # Process each metric's analysis
    for metric, results in analysis_results.items():
        if not isinstance(results, dict):
            continue
            
        # Process agreement results first (kept separate from other metrics)
        if metric.startswith("agreement_"):
            # Process agreement results
            framework = metric.replace("agreement_", "")
            lines.append(f"## Inter-Model Agreement for {framework.replace('_', ' ').title()}")
            
            if "overall" in results:
                overall = results["overall"]
                if isinstance(overall, dict) and "error" not in overall:
                    kappa = overall.get("average_kappa")
                    level = overall.get("agreement_level")
                    if kappa is not None and level is not None:
                        lines.append(f"- Overall Agreement: κ = {kappa:.3f} ({level})")
                        # Add to key findings if substantial or better
                        if level in ["substantial", "almost perfect"]:
                            key_findings.append(f"High inter-model agreement for {framework.replace('_', ' ').title()} (κ = {kappa:.3f}, {level})")
            
            lines.append("")
        else:
            # Determine if this is an accuracy/gold standard metric
            is_accuracy_metric = (
                metric == "framework_agreement" or 
                metric == "semantic_similarity" or
                metric.startswith("framework_accuracy_")
            )
            
            # Create a more readable metric name
            readable_metric = metric
            if metric.startswith("framework_accuracy_"):
                readable_metric = f"Accuracy: {metric.replace('framework_accuracy_', '')}"
            elif metric == "framework_agreement":
                readable_metric = "Framework Agreement"
            elif metric == "semantic_similarity":
                readable_metric = "Semantic Similarity"
            
            readable_metric = readable_metric.replace("_", " ").title()
            
            # Process metric analysis results
            lines.append(f"## {readable_metric}")
            
            # If this is an accuracy metric, mark it as important for the paper
            if is_accuracy_metric:
                lines.append("*This is a key accuracy metric for evaluating model performance against gold standards.*")
                lines.append("")
            
            # ANOVA results
            if "anova" in results:
                anova = results["anova"]
                if isinstance(anova, dict) and "error" not in anova:
                    f_stat = anova.get("f_statistic")
                    p_val = anova.get("p_value")
                    sig = anova.get("significant")
                    groups = anova.get("groups")
                    
                    if f_stat is not None and p_val is not None:
                        sig_text = "significant" if sig else "not significant"
                        lines.append(f"- One-way ANOVA: F = {f_stat:.3f}, p = {p_val:.4f} ({sig_text})")
                        
                        # Add to key findings if significant and it's an accuracy metric
                        if sig and is_accuracy_metric:
                            key_findings.append(f"Significant difference between models for {readable_metric} (F = {f_stat:.3f}, p = {p_val:.4f})")
                        
                        # Include group names if available
                        if groups and isinstance(groups, list) and len(groups) > 0:
                            group_text = ", ".join([str(g) for g in groups])
                            lines.append(f"  - Compared models: {group_text}")
            
            # Effect size results
            if "effect_size" in results:
                effect = results["effect_size"]
                if isinstance(effect, dict) and "error" not in effect and "comparisons" in effect:
                    comparisons = effect.get("comparisons")
                    if comparisons and isinstance(comparisons, dict):
                        lines.append("- Effect Sizes (Cohen's d):")
                        
                        # Keep track of notable effect sizes
                        large_effects = []
                        for comparison, values in comparisons.items():
                            if not isinstance(values, dict):
                                continue
                                
                            d = values.get("cohen_d")
                            level = values.get("effect_size")
                            if d is not None and level:
                                lines.append(f"  - {comparison}: d = {d:.3f} ({level})")
                                
                                # Track large effect sizes for key findings
                                if level == "large" and is_accuracy_metric:
                                    large_effects.append(f"{comparison} (d = {d:.3f})")
                        
                        # Add large effects to key findings
                        if large_effects and is_accuracy_metric:
                            key_findings.append(f"Large effect sizes for {readable_metric}: {', '.join(large_effects)}")
            
            # Confidence intervals
            if "confidence_intervals" in results:
                ci = results["confidence_intervals"]
                if isinstance(ci, dict) and "error" not in ci and "intervals" in ci:
                    intervals = ci.get("intervals")
                    if intervals and isinstance(intervals, dict):
                        lines.append("- 95% Confidence Intervals:")
                        
                        # Find the model with the highest mean for key findings
                        best_model = None
                        best_mean = -float('inf')
                        
                        for model, values in intervals.items():
                            if not isinstance(values, dict) or "error" in values:
                                continue
                                
                            mean = values.get("mean")
                            lower = values.get("lower_ci")
                            upper = values.get("upper_ci")
                            if mean is not None and lower is not None and upper is not None:
                                lines.append(f"  - {model}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
                                
                                # Track the model with highest mean for accuracy metrics
                                if is_accuracy_metric and mean > best_mean:
                                    best_mean = mean
                                    best_model = model
                        
                        # Add best model to key findings
                        if best_model and is_accuracy_metric and best_mean > 0:
                            key_findings.append(f"Best model for {readable_metric}: {best_model} ({best_mean:.3f})")
            
            lines.append("")
    
    # Insert key findings at the top of the document
    if key_findings:
        summary_lines = ["### Most Important Findings", ""]
        for i, finding in enumerate(key_findings):
            summary_lines.append(f"{i+1}. {finding}")
        summary_lines.append("")
        
        # Insert after the Key Findings Summary header
        insert_position = 4  # After title, blank line, section header, and description
        lines[insert_position:insert_position] = summary_lines
    
    # Add raw visualization data section before recommendations
    lines.append("## Raw Data for Visualizations")
    lines.append("This section provides the raw data used in generating visualizations for the paper.")
    lines.append("")
    
    # Format the accuracy metrics data for gold standard accuracy plots
    lines.append("### Data for Gold Standard Accuracy Plots")
    lines.append("```")
    lines.append("Framework                     | Model Name                         | Accuracy | Lower CI | Upper CI")
    lines.append("---------------------------------------------------------------------------------------------")
    
    # Add data for each framework accuracy metric from the confidence intervals
    for metric, results in analysis_results.items():
        if not isinstance(results, dict):
            continue
            
        if metric.startswith("framework_accuracy_"):
            framework_name = metric.replace("framework_accuracy_", "")
            readable_framework = framework_name.replace("_", " ").title()
            
            if "confidence_intervals" in results:
                ci = results["confidence_intervals"]
                if isinstance(ci, dict) and "intervals" in ci:
                    intervals = ci.get("intervals")
                    if intervals and isinstance(intervals, dict):
                        for model, values in intervals.items():
                            if not isinstance(values, dict):
                                continue
                                
                            mean = values.get("mean", 0.0)
                            lower = values.get("lower_ci", 0.0)
                            upper = values.get("upper_ci", 0.0)
                            
                            lines.append(f"{readable_framework.ljust(30)} | {model.ljust(35)} | {mean:.3f}   | {lower:.3f}   | {upper:.3f}")
    
    lines.append("```")
    lines.append("")
    
    # Add data for framework agreement and semantic similarity
    for metric_name, display_name in [
        ("framework_agreement", "Framework Agreement"), 
        ("semantic_similarity", "Semantic Similarity")
    ]:
        if metric_name in analysis_results:
            results = analysis_results[metric_name]
            if "confidence_intervals" in results:
                ci = results["confidence_intervals"]
                if isinstance(ci, dict) and "intervals" in ci:
                    lines.append(f"### Data for {display_name} Plot")
                    lines.append("```")
                    lines.append(f"Model Name                         | {display_name} | Lower CI | Upper CI")
                    lines.append("--------------------------------------------------------------------")
                    
                    intervals = ci.get("intervals")
                    if intervals and isinstance(intervals, dict):
                        for model, values in intervals.items():
                            if not isinstance(values, dict):
                                continue
                                
                            mean = values.get("mean", 0.0)
                            lower = values.get("lower_ci", 0.0)
                            upper = values.get("upper_ci", 0.0)
                            
                            lines.append(f"{model.ljust(35)} | {mean:.3f}        | {lower:.3f}   | {upper:.3f}")
                    
                    lines.append("```")
                    lines.append("")
    
    # Add data for correlation matrix if available
    has_correlation_data = False
    for metric, results in analysis_results.items():
        if isinstance(results, dict) and "effect_size" in results:
            has_correlation_data = True
            break
            
    if has_correlation_data:
        lines.append("### Data for Correlation Matrix")
        lines.append("The correlation matrix is constructed using the following metrics:")
        lines.append("- Framework Agreement")
        lines.append("- Semantic Similarity")
        lines.append("- Framework-specific Accuracy")
        lines.append("- Average Framework Confidence")
        lines.append("- Reasoning Count")
        lines.append("")
        lines.append("*Detailed correlation values are available in the statistical_analysis.json file.*")
        lines.append("")
    
    # Add data for confidence interval plots
    lines.append("### Data for Confidence Interval Plots")
    lines.append("Confidence interval plots use the same data as shown in the 95% Confidence Intervals sections above.")
    lines.append("The plots visualize the mean values with error bars representing the 95% confidence intervals.")
    lines.append("")
    
    # Add data for version comparison plots if we have multiple versions
    # Check if we have version data by looking for patterns in the metric names
    has_version_data = False
    version_metrics = {}
    for metric, results in analysis_results.items():
        if isinstance(results, dict) and "confidence_intervals" in results:
            ci = results["confidence_intervals"]
            if isinstance(ci, dict) and "intervals" in ci:
                intervals = ci.get("intervals")
                if intervals and isinstance(intervals, dict):
                    for model in intervals.keys():
                        if "_v" in model.lower() or "version" in model.lower():
                            has_version_data = True
                            if metric not in version_metrics:
                                version_metrics[metric] = []
                            version_metrics[metric].append(model)
    
    if has_version_data:
        lines.append("### Data for Version Comparison Plots")
        lines.append("Version comparison plots show how model performance changes across different versions.")
        lines.append("The plots use the following data:")
        lines.append("")
        
        for metric, models in version_metrics.items():
            readable_metric = metric
            if metric.startswith("framework_accuracy_"):
                readable_metric = f"Accuracy: {metric.replace('framework_accuracy_', '')}"
            elif metric == "framework_agreement":
                readable_metric = "Framework Agreement"
            elif metric == "semantic_similarity":
                readable_metric = "Semantic Similarity"
            
            readable_metric = readable_metric.replace("_", " ").title()
            
            lines.append(f"#### {readable_metric}")
            lines.append("```")
            lines.append("Model/Version                     | Value   | Lower CI | Upper CI")
            lines.append("------------------------------------------------------------------")
            
            if metric in analysis_results and "confidence_intervals" in analysis_results[metric]:
                ci = analysis_results[metric]["confidence_intervals"]
                if isinstance(ci, dict) and "intervals" in ci:
                    intervals = ci.get("intervals")
                    if intervals and isinstance(intervals, dict):
                        for model in models:
                            if model in intervals:
                                values = intervals[model]
                                if not isinstance(values, dict):
                                    continue
                                    
                                mean = values.get("mean", 0.0)
                                lower = values.get("lower_ci", 0.0)
                                upper = values.get("upper_ci", 0.0)
                                
                                lines.append(f"{model.ljust(35)} | {mean:.3f}  | {lower:.3f}   | {upper:.3f}")
            
            lines.append("```")
            lines.append("")
    
    # Add data for detailed prediction analysis
    lines.append("### Data for Detailed Prediction Analysis Heatmaps")
    lines.append("The detailed prediction analysis heatmaps visualize the following data:")
    lines.append("- Framework accuracy per test case")
    lines.append("- Model performance across different frameworks")
    lines.append("- Patterns of errors in model predictions")
    lines.append("")
    lines.append("*This visualization uses the individual case-level data which is too extensive to include here.*")
    lines.append("*Refer to the evaluation_results.json file for the complete case-by-case data.*")
    lines.append("")
    
    # Add recommendations section at the end
    lines.append("## Recommendations for Paper")
    lines.append("Based on the statistical analysis, consider highlighting the following in your paper:")
    lines.append("")
    lines.append("1. **Explicit comparison to gold standards** - This is the most important aspect for evaluating model performance.")
    lines.append("2. **Statistical significance of model differences** - Emphasize where models show significant differences.")
    lines.append("3. **Effect sizes between models** - These show practical significance beyond statistical significance.")
    lines.append("4. **Confidence intervals** - These provide estimates of precision and help with reproducibility claims.")
    lines.append("5. **Inter-model agreement** - This demonstrates consistency across models when evaluating the same profiles.")
    
    return "\n".join(lines)

def export_results(results: List[Dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    """Export evaluation results to files.
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save results
        
    Returns:
        DataFrame with summarized results
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save raw results
        save_results_to_json(results, output_dir / "evaluation_results.json")
        
        # Create summary DataFrame
        df = create_summary_dataframe(results)
        
        # Skip if DataFrame is empty
        if df.empty:
            logger.warning("No valid data for summary DataFrame")
            return df
            
        # Save summary to CSV
        save_results_to_csv(df, output_dir / "evaluation_summary.csv")
        
        # Perform statistical analysis
        if STATISTICAL_ANALYSIS_AVAILABLE and not df.empty:
            try:
                perform_statistical_analysis(df, output_dir)
            except Exception as e:
                logger.error(f"Error during statistical analysis: {e}")
        
        return df
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return pd.DataFrame()