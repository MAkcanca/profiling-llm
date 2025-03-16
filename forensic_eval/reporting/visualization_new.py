"""Simplified visualization utilities focused on grouped bar charts."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from statsmodels.stats.anova import AnovaRM

# Set up logger
logger = logging.getLogger(__name__)

# Define visualization constants for a consistent style
SMALL_SIZE = 9
MEDIUM_SIZE = 11
LARGE_SIZE = 14
TITLE_SIZE = 16

# Set publication-ready styling
plt.style.use(['seaborn-v0_8-whitegrid', 'seaborn-v0_8-paper'])

# Figure settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = SMALL_SIZE
plt.rcParams['axes.titlesize'] = LARGE_SIZE
plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE
plt.rcParams['legend.fontsize'] = SMALL_SIZE

# Color palette (optimized for scientific papers)
PAPER_PALETTE = sns.color_palette([
    "#4878D0", "#EE854A", "#6ACC64", "#D65F5F",
    "#956CB4", "#8C613C", "#DC7EC0", "#797979"
])

def create_horizontal_grouped_bar_chart(results: List[Dict], plots_dir: Path, metrics: Optional[List[str]] = None) -> None:
    """Create a horizontal grouped bar chart comparing models across selected metrics.
    
    Args:
        results: List of evaluation result dictionaries (already loaded)
        plots_dir: Directory to save the plot
        metrics: Optional list of metrics to include in chart (defaults to framework accuracy metrics if None)
    """
    try:
        # Ensure plots directory exists
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Flatten results for visualization
        flattened_results = []
        for result in results:
            if not isinstance(result, dict):
                continue
                
            # Extract basic information
            flat_result = {
                "test_case": str(result.get("test_case", "")),
                "model": str(result.get("model", "")),
                "version": str(result.get("version", "latest"))
            }
            
            # Extract gold standard metrics
            gold_standard = result.get("gold_standard")
            if gold_standard and isinstance(gold_standard, dict):
                # Add framework agreement if available
                framework_agreement = gold_standard.get("framework_agreement")
                if framework_agreement is not None and isinstance(framework_agreement, (int, float)):
                    flat_result["framework_agreement"] = framework_agreement
                    
                # Add semantic similarity if available
                sem_sim = gold_standard.get("semantic_similarity")
                if sem_sim is not None and isinstance(sem_sim, (int, float)):
                    flat_result["semantic_similarity"] = sem_sim
                
                # Add accuracy metrics if available
                accuracy = gold_standard.get("accuracy")
                if accuracy and isinstance(accuracy, dict):
                    for fw_name, acc_value in accuracy.items():
                        if isinstance(acc_value, (int, float)):
                            flat_result[f"framework_accuracy_{fw_name}"] = acc_value
            
            flattened_results.append(flat_result)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_results)
        
        # Skip if DataFrame is empty
        if df.empty:
            logger.warning("No valid data for visualization")
            return
            
        # If no metrics specified, use framework accuracy metrics by default
        if metrics is None:
            metrics = [col for col in df.columns if col.startswith("framework_accuracy_")]
            # Add standard metrics if available
            standard_metrics = ["framework_agreement", "semantic_similarity"]
            for metric in standard_metrics:
                if metric in df.columns:
                    metrics.append(metric)
        
        # Validate that we have the requested metrics
        valid_metrics = [m for m in metrics if m in df.columns]
        
        if not valid_metrics:
            logger.warning("No valid metrics found for visualization")
            return
            
        # Group by model and calculate mean for each metric
        aggregated_df = df.groupby('model')[valid_metrics].mean().reset_index()
        
        # Sort models by overall performance
        aggregated_df['avg_score'] = aggregated_df[valid_metrics].mean(axis=1)
        aggregated_df = aggregated_df.sort_values('avg_score', ascending=False)
        
        # Get the model order for consistent display
        model_order = aggregated_df['model'].tolist()
        
        # Melt the dataframe for seaborn plotting
        melted_df = pd.melt(
            aggregated_df, 
            id_vars=['model'], 
            value_vars=valid_metrics,
            var_name='Metric', 
            value_name='Score'
        )
        
        # Format metric names for better readability
        melted_df['Metric'] = melted_df['Metric'].apply(lambda m: 
            m.replace("framework_accuracy_", "")
            .replace("framework_agreement", "Framework Agreement")
            .replace("semantic_similarity", "Semantic Similarity")
            .replace("_", " ")
            .title()
        )
        metric_abbreviations = {
            "Narrative Action System": "NAS",
            "Sexual Behavioral Analysis": "SBA",
            "Spatial Behavioral Analysis": "SPBA",
            "Sexual Homicide Pathways Analysis": "SHPA",
        }
        # Apply abbreviations if we have any
        if metric_abbreviations:
            melted_df['Metric'] = melted_df['Metric'].apply(
                lambda m: metric_abbreviations.get(m, m)
            )

        
        # Abbreviate long model names for better display
        model_abbreviations = {
            "GPT-4.5-Preview": "GPT-4.5",
            "Claude-3.7-Sonnet-Thinking": "Claude-3.7-Think",
            "Llama-3.3-70B-Instruct": "Llama-3.3",
            "Gemini-2.0-Flash-Thinking-Exp0121": "Gemini-2.0-Think",
        }
        
        melted_df['display_model'] = melted_df['model'].apply(
            lambda m: model_abbreviations.get(m, m)
        )
        
        # Create the horizontal grouped bar chart
        # Calculate appropriate figure dimensions based on data size
        n_models = len(model_order)
        n_metrics = len(valid_metrics)
        horizontal_fig_width = min(8, max(5, 3 + 0.3 * n_metrics))
        horizontal_fig_height = min(9, max(3, 1.5 + 0.5 * n_models))

        fig, ax = plt.subplots(figsize=(horizontal_fig_width, horizontal_fig_height), dpi=300)
        
        # Create the plot
        ax = sns.barplot(
            data=melted_df,
            y='display_model',
            x='Score',
            hue='Metric',
            palette=PAPER_PALETTE[:len(valid_metrics)],
            orient='h',
            width=0.8,  # Slightly narrower bars for clarity
            saturation=0.9  # Better visibility in print
        )
        
        # Add value labels to the bars
        for i, bar in enumerate(ax.containers):
            ax.bar_label(bar, fmt='%.2f', padding=3, fontsize=SMALL_SIZE-1)
        
        # Set title and labels
        plt.title("Model Performance Comparison", fontweight="bold", fontsize=TITLE_SIZE, pad=20)
        plt.xlabel("Score", fontsize=MEDIUM_SIZE)
        plt.ylabel("")  # No y-axis label needed
        
        # Set x-axis limits appropriate for accuracy metrics
        plt.xlim(0, min(1.05, melted_df['Score'].max() * 1.1))
        
        # Add grid for readability
        plt.grid(axis='x', linestyle=':', alpha=0.3, linewidth=0.6)
        
        # Clean up spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Add legend with good positioning
        plt.legend(
            loc='lower right',
            frameon=False,
            fontsize=SMALL_SIZE-3,
            title_fontsize=SMALL_SIZE+1
        )
        
        # Ensure tight layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / "model_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created horizontal grouped bar chart at {plots_dir / 'model_metrics_comparison.png'}")
        
    except Exception as e:
        logger.error(f"Error creating horizontal grouped bar chart: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_markdown_report(results: List[Dict], plots_dir: Path) -> None:
    """Generate a structured markdown report with all statistics and tables for LLM consumption.
    
    This function extracts all relevant data from the results and formats it into a comprehensive
    markdown file that can be used by an LLM to write a scientific paper.
    
    Args:
        results: List of evaluation result dictionaries (already loaded)
        plots_dir: Directory to save the markdown report
    """
    try:
        # Ensure plots directory exists
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a markdown file for the report
        md_file_path = plots_dir / "llm_research_report.md"
        
        # Flatten results for analysis
        flattened_results = []
        for result in results:
            if not isinstance(result, dict):
                continue
                
            # Extract basic information
            flat_result = {
                "test_case": str(result.get("test_case", "")),
                "model": str(result.get("model", "")),
                "version": str(result.get("version", "latest"))
            }
            
            # Add reasoning metrics if available
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
            
            # Add framework metrics if available
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
                
                # Add framework classifications
                classifications = framework.get("classifications")
                if classifications and isinstance(classifications, dict):
                    for fw_name, classification in classifications.items():
                        flat_result[f"classification_{fw_name}"] = str(classification)
            
            # Add gold standard metrics if available
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
                
                # Add framework match details
                framework_matches = gold_standard.get("framework_matches")
                if framework_matches and isinstance(framework_matches, dict):
                    for fw_name, match_info in framework_matches.items():
                        if isinstance(match_info, dict):
                            # Save match status (True/False)
                            flat_result[f"match_{fw_name}"] = match_info.get("match", False)
                            # Save the gold standard value
                            flat_result[f"gold_{fw_name}"] = str(match_info.get("gold", ""))
                
                # Add framework contribution metrics if available
                framework_contribution = gold_standard.get("framework_contribution")
                if framework_contribution and isinstance(framework_contribution, dict):
                    # Add individual contributions for each framework
                    individual_contributions = framework_contribution.get("individual_contributions")
                    if individual_contributions and isinstance(individual_contributions, dict):
                        for fw_name, contribution in individual_contributions.items():
                            if isinstance(contribution, (int, float)):
                                flat_result[f"framework_contribution_{fw_name}"] = contribution
                    
                    # Add relative importance for each framework
                    relative_importance = framework_contribution.get("relative_importance")
                    if relative_importance and isinstance(relative_importance, dict):
                        for fw_name, importance in relative_importance.items():
                            if isinstance(importance, (int, float)):
                                flat_result[f"framework_importance_{fw_name}"] = importance
                    
                    # Add most influential frameworks
                    most_influential = framework_contribution.get("most_influential_frameworks")
                    if most_influential and isinstance(most_influential, list):
                        flat_result["most_influential_frameworks"] = ",".join(most_influential)
            
            flattened_results.append(flat_result)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(flattened_results)
        
        # Skip if DataFrame is empty
        if df.empty:
            logger.warning("No valid data for markdown report")
            return
        
        # Begin writing markdown report
        with open(md_file_path, 'w', encoding='utf-8') as md_file:
            # Title and introduction
            md_file.write("# Forensic LLM Analysis Research Report\n\n")
            md_file.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n")
            
            md_file.write("## Executive Summary\n\n")
            md_file.write("This report contains detailed statistics and analyses of different language models' performance ")
            md_file.write("on forensic analysis tasks. The data is structured to facilitate the writing of a scientific paper ")
            md_file.write("based on these results.\n\n")
            
            # Add methodological notes and limitations
            md_file.write("### Methodological Notes\n\n")
            md_file.write("#### Statistical Methodology\n\n")
            md_file.write("- **Multiple Comparisons Correction**: False Discovery Rate (FDR) correction using the Benjamini-Hochberg procedure has been applied to control the expected proportion of false positives in all pairwise comparisons. ")
            md_file.write("This procedure adjusts p-values to control the proportion of Type I errors (false positives) when conducting multiple statistical tests, offering greater statistical power than family-wise error rate methods.\n\n")
            
            md_file.write("#### Limitations\n\n")
            md_file.write("- **Sample Size**: This study analyzed a limited set of test cases. Results should be interpreted as preliminary findings that warrant further investigation with a larger dataset.\n\n")
            md_file.write("- **Framework Specificity**: Performance metrics may be specific to the forensic frameworks used in this evaluation and may not generalize to all forensic analysis tasks.\n\n")
            
            # Dataset overview
            md_file.write("## Dataset Overview\n\n")
            
            # Count of test cases and models
            test_cases = df['test_case'].nunique()
            models = df['model'].unique().tolist()
            model_count = len(models)
            
            md_file.write(f"- **Total test cases evaluated**: {test_cases}\n")
            md_file.write(f"- **Number of models evaluated**: {model_count}\n")
            md_file.write(f"- **Models included in analysis**: {', '.join(models)}\n\n")
            
            # Model performance summary
            md_file.write("## Model Performance Summary\n\n")
            
            # Identify key metrics for the report
            accuracy_metrics = [col for col in df.columns if col.startswith("framework_accuracy_") and not df[col].isna().all()]
            standard_metrics = ["framework_agreement", "semantic_similarity"]
            valid_metrics = [m for m in standard_metrics if m in df.columns and not df[m].isna().all()] + accuracy_metrics
            
            # Create a summary table of model performance across metrics
            if valid_metrics:
                md_file.write("### Overall Performance Metrics\n\n")
                
                # Group by model and calculate mean for each metric
                model_performance = df.groupby('model')[valid_metrics].mean().reset_index()
                
                # Sort models by average score across all metrics
                model_performance['avg_score'] = model_performance[valid_metrics].mean(axis=1)
                model_performance = model_performance.sort_values('avg_score', ascending=False)
                
                # Format metric names for better readability
                readable_metrics = {}
                for metric in valid_metrics:
                    if metric.startswith("framework_accuracy_"):
                        readable_metrics[metric] = "Accuracy: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                    elif metric == "framework_agreement":
                        readable_metrics[metric] = "Framework Agreement"
                    elif metric == "semantic_similarity":
                        readable_metrics[metric] = "Semantic Similarity"
                    else:
                        readable_metrics[metric] = metric.replace("_", " ").title()
                
                # Create markdown table header
                md_file.write("| Model | " + " | ".join([readable_metrics[m] for m in valid_metrics]) + " | Average |\n")
                md_file.write("|-------|" + "|".join(["-" * (len(readable_metrics[m]) + 2) for m in valid_metrics]) + "-|-------|\n")
                
                # Add rows for each model
                for _, row in model_performance.iterrows():
                    model_name = row['model']
                    metrics_values = [f"{row[m]:.3f}" for m in valid_metrics]
                    avg_score = row['avg_score']
                    md_file.write(f"| {model_name} | " + " | ".join(metrics_values) + f" | {avg_score:.3f} |\n")
                
                md_file.write("\n")
            
            # Detailed framework-specific analysis
            md_file.write("## Framework-Specific Analysis\n\n")
            
            # Get framework-specific columns
            framework_prefixes = ["framework_accuracy_", "framework_confidence_", "framework_completeness_"]
            
            # Extract all unique frameworks from the data
            frameworks = set()
            for prefix in framework_prefixes:
                frameworks.update([col.replace(prefix, "") for col in df.columns if col.startswith(prefix)])
            
            # Framework contribution analysis section
            md_file.write("## Framework Contribution Analysis\n\n")
            md_file.write("This section analyzes how much each framework contributes to the overall accuracy of profiles.\n\n")
            
            # Extract contribution metrics
            contribution_frameworks = set([col.replace("framework_contribution_", "") for col in df.columns if col.startswith("framework_contribution_")])
            importance_frameworks = set([col.replace("framework_importance_", "") for col in df.columns if col.startswith("framework_importance_")])
            
            if contribution_frameworks:
                md_file.write("### Individual Framework Contributions\n\n")
                md_file.write("This table shows how much each framework contributes to the overall accuracy. ")
                md_file.write("Positive values indicate correct classifications that improve overall accuracy, ")
                md_file.write("while negative values represent potential contributions if corrected.\n\n")
                
                # Calculate average contribution across all models and test cases
                contribution_cols = [f"framework_contribution_{fw}" for fw in contribution_frameworks]
                avg_contributions = df[contribution_cols].mean().reset_index()
                avg_contributions.columns = ['Framework', 'Average Contribution']
                
                # Sort by absolute contribution (most influential first)
                avg_contributions['Abs Contribution'] = avg_contributions['Average Contribution'].abs()
                avg_contributions = avg_contributions.sort_values('Abs Contribution', ascending=False)
                
                # Create markdown table
                md_file.write("| Framework | Average Contribution | Absolute Contribution |\n")
                md_file.write("|-----------|----------------------|----------------------|\n")
                
                for _, row in avg_contributions.iterrows():
                    framework = row['Framework'].replace("framework_contribution_", "").replace("_", " ").title()
                    avg_contrib = row['Average Contribution']
                    abs_contrib = row['Abs Contribution']
                    md_file.write(f"| {framework} | {avg_contrib:.4f} | {abs_contrib:.4f} |\n")
                
                md_file.write("\n")
            
            if importance_frameworks:
                md_file.write("### Relative Framework Importance\n\n")
                md_file.write("This table shows the relative importance of each framework (normalized contribution). ")
                md_file.write("Higher values indicate frameworks that have a greater impact on overall profile accuracy.\n\n")
                
                # Calculate average importance across all models and test cases
                importance_cols = [f"framework_importance_{fw}" for fw in importance_frameworks]
                avg_importance = df[importance_cols].mean().reset_index()
                avg_importance.columns = ['Framework', 'Relative Importance']
                
                # Sort by importance (most important first)
                avg_importance = avg_importance.sort_values('Relative Importance', ascending=False)
                
                # Create markdown table
                md_file.write("| Framework | Relative Importance |\n")
                md_file.write("|-----------|---------------------|\n")
                
                for _, row in avg_importance.iterrows():
                    framework = row['Framework'].replace("framework_importance_", "").replace("_", " ").title()
                    importance = row['Relative Importance']
                    md_file.write(f"| {framework} | {importance:.4f} |\n")
                
                md_file.write("\n")
            
            # Most influential frameworks analysis
            if "most_influential_frameworks" in df.columns:
                md_file.write("### Most Influential Frameworks\n\n")
                md_file.write("This analysis identifies which frameworks are most frequently identified as the most influential ")
                md_file.write("for overall profile accuracy.\n\n")
                
                # Extract most influential frameworks and count occurrences
                influential_counts = {}
                
                for _, row in df.iterrows():
                    if pd.notna(row.get('most_influential_frameworks')):
                        frameworks_list = row['most_influential_frameworks'].split(',')
                        for fw in frameworks_list:
                            influential_counts[fw] = influential_counts.get(fw, 0) + 1
                
                if influential_counts:
                    # Convert to dataframe for easier manipulation
                    influential_df = pd.DataFrame({
                        'Framework': list(influential_counts.keys()),
                        'Count': list(influential_counts.values())
                    })
                    
                    # Sort by count (most frequent first)
                    influential_df = influential_df.sort_values('Count', ascending=False)
                    
                    # Calculate percentage
                    total_cases = len(df)
                    influential_df['Percentage'] = influential_df['Count'] / total_cases * 100
                    
                    # Create markdown table
                    md_file.write("| Framework | Count | % of Cases |\n")
                    md_file.write("|-----------|-------|------------|\n")
                    
                    for _, row in influential_df.iterrows():
                        framework = row['Framework'].replace("_", " ").title()
                        count = row['Count']
                        percentage = row['Percentage']
                        md_file.write(f"| {framework} | {count} | {percentage:.1f}% |\n")
                    
                    md_file.write("\n")
            
            # For each framework, provide detailed analysis
            for framework in frameworks:
                md_file.write(f"### {framework.replace('_', ' ').title()} Framework\n\n")
                
                # Framework accuracy analysis
                accuracy_col = f"framework_accuracy_{framework}"
                if accuracy_col in df.columns and not df[accuracy_col].isna().all():
                    md_file.write(f"#### Accuracy Analysis for {framework.replace('_', ' ').title()}\n\n")
                    
                    # Calculate model-specific accuracy rates
                    model_accuracy = df.groupby('model')[accuracy_col].agg(['mean', 'count']).reset_index()
                    model_accuracy = model_accuracy.sort_values('mean', ascending=False)
                    
                    # Create markdown table
                    md_file.write("| Model | Accuracy | Sample Count |\n")
                    md_file.write("|-------|----------|-------------|\n")
                    
                    for _, row in model_accuracy.iterrows():
                        md_file.write(f"| {row['model']} | {row['mean']:.3f} | {int(row['count'])} |\n")
                    
                    md_file.write("\n")
                
                # Framework confidence analysis
                confidence_col = f"framework_confidence_{framework}"
                if confidence_col in df.columns and not df[confidence_col].isna().all():
                    md_file.write(f"#### Confidence Analysis for {framework.replace('_', ' ').title()}\n\n")
                    
                    # Calculate model-specific confidence rates
                    model_confidence = df.groupby('model')[confidence_col].mean().reset_index()
                    model_confidence = model_confidence.sort_values(confidence_col, ascending=False)
                    
                    # Create markdown table
                    md_file.write("| Model | Confidence |\n")
                    md_file.write("|-------|------------|\n")
                    
                    for _, row in model_confidence.iterrows():
                        md_file.write(f"| {row['model']} | {row[confidence_col]:.3f} |\n")
                    
                    md_file.write("\n")
                
                # Framework completeness analysis
                completeness_col = f"framework_completeness_{framework}"
                if completeness_col in df.columns and not df[completeness_col].isna().all():
                    md_file.write(f"#### Completeness Analysis for {framework.replace('_', ' ').title()}\n\n")
                    
                    # Calculate model-specific completeness rates
                    model_completeness = df.groupby('model')[completeness_col].mean().reset_index()
                    model_completeness = model_completeness.sort_values(completeness_col, ascending=False)
                    
                    # Create markdown table
                    md_file.write("| Model | Completeness |\n")
                    md_file.write("|-------|-------------|\n")
                    
                    for _, row in model_completeness.iterrows():
                        md_file.write(f"| {row['model']} | {row[completeness_col]:.3f} |\n")
                    
                    md_file.write("\n")
                
                # Framework classification analysis
                classification_col = f"classification_{framework}"
                gold_col = f"gold_{framework}"
                match_col = f"match_{framework}"
                
                if all(col in df.columns for col in [classification_col, gold_col, match_col]):
                    md_file.write(f"#### Classification Analysis for {framework.replace('_', ' ').title()}\n\n")
                    
                    # Filter out rows where gold standard is null, None, or "null"
                    framework_df = df[~df[gold_col].isin(["null", "None", "", "Null"])]
                    framework_df = framework_df[~framework_df[gold_col].isna()]
                    
                    if not framework_df.empty:
                        # Calculate success rates by model
                        model_success = framework_df.groupby('model')[match_col].mean().reset_index()
                        model_success = model_success.sort_values(match_col, ascending=False)
                        
                        # Create markdown table
                        md_file.write("| Model | Success Rate | Correct | Total |\n")
                        md_file.write("|-------|-------------|---------|-------|\n")
                        
                        for _, row in model_success.iterrows():
                            model_data = framework_df[framework_df['model'] == row['model']]
                            correct = model_data[match_col].sum()
                            total = len(model_data)
                            
                            md_file.write(f"| {row['model']} | {row[match_col]:.3f} | {int(correct)} | {total} |\n")
                        
                        md_file.write("\n")
                        
                        # Case-specific analysis
                        md_file.write("#### Case-by-Case Analysis\n\n")
                        
                        # Calculate success rates by test case
                        case_success = framework_df.groupby('test_case')[match_col].mean().reset_index()
                        case_success = case_success.sort_values(match_col, ascending=False)
                        
                        # Create markdown table
                        md_file.write("| Test Case | Gold Standard | Success Rate | Models with Correct Prediction |\n")
                        md_file.write("|-----------|---------------|-------------|---------------------------------|\n")
                        
                        for _, row in case_success.iterrows():
                            case_data = framework_df[framework_df['test_case'] == row['test_case']]
                            gold = case_data[gold_col].iloc[0]
                            correct_models = case_data[case_data[match_col] > 0]['model'].tolist()
                            correct_list = ", ".join(correct_models)
                            
                            md_file.write(f"| {row['test_case']} | {gold} | {row[match_col]:.3f} | {correct_list} |\n")
                        
                        md_file.write("\n")
                        
                        # Detailed classification matrix
                        md_file.write("#### Detailed Classification Matrix\n\n")
                        
                        # Create a pivot table showing predictions by model and case
                        classification_matrix = pd.pivot_table(
                            framework_df, 
                            values=classification_col,
                            index='test_case',
                            columns='model',
                            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else ""
                        )
                        
                        # Add gold standard column
                        classification_matrix['Gold Standard'] = framework_df.groupby('test_case')[gold_col].first()
                        
                        # Convert to markdown
                        md_file.write(classification_matrix.to_markdown() + "\n\n")
            
            # Reasoning analysis
            reasoning_cols = [col for col in df.columns if col.startswith("reasoning_") and not df[col].isna().all()]
            
            if reasoning_cols:
                md_file.write("## Reasoning Analysis\n\n")
                
                if "reasoning_count" in reasoning_cols:
                    md_file.write("### Overall Reasoning Count by Model\n\n")
                    
                    # Calculate average reasoning count by model
                    model_reasoning = df.groupby('model')["reasoning_count"].mean().reset_index()
                    model_reasoning = model_reasoning.sort_values("reasoning_count", ascending=False)
                    
                    # Create markdown table
                    md_file.write("| Model | Average Reasoning Count |\n")
                    md_file.write("|-------|-----------------------|\n")
                    
                    for _, row in model_reasoning.iterrows():
                        md_file.write(f"| {row['model']} | {row['reasoning_count']:.2f} |\n")
                    
                    md_file.write("\n")
                
                # Detailed reasoning by section if available
                section_cols = [col for col in reasoning_cols if col != "reasoning_count"]
                
                if section_cols:
                    md_file.write("### Detailed Reasoning by Section\n\n")
                    
                    # Group by model and calculate mean for each section
                    section_reasoning = df.groupby('model')[section_cols].mean().reset_index()
                    
                    # Calculate total reasoning across sections for sorting
                    section_reasoning['total'] = section_reasoning[section_cols].sum(axis=1)
                    section_reasoning = section_reasoning.sort_values('total', ascending=False)
                    
                    # Format section names for better readability
                    readable_sections = {}
                    for col in section_cols:
                        section_name = col.replace("reasoning_", "").replace("_", " ").title()
                        readable_sections[col] = section_name
                    
                    # Create markdown table
                    md_file.write("| Model | " + " | ".join([readable_sections[col] for col in section_cols]) + " | Total |\n")
                    md_file.write("|" + "-" * 7 + "|" + "".join(["-" * (len(readable_sections[col]) + 2) + "|" for col in section_cols]) + "-" * 7 + "|\n")
                    
                    for _, row in section_reasoning.iterrows():
                        model_name = row['model']
                        section_values = [f"{row[col]:.2f}" for col in section_cols]
                        total = row['total']
                        md_file.write(f"| {model_name} | " + " | ".join(section_values) + f" | {total:.2f} |\n")
                    
                    md_file.write("\n")
            
            # Correlation analysis
            md_file.write("## Correlation Analysis\n\n")
            
            # Define essential metrics for correlation
            essential_metrics = [
                "reasoning_count", 
                "avg_framework_confidence", 
                "framework_agreement", 
                "semantic_similarity"
            ] + accuracy_metrics
            
            # Filter to metrics that exist in the dataframe
            valid_metrics = [m for m in essential_metrics if m in df.columns and not df[m].isna().all()]
            
            if len(valid_metrics) >= 2:
                # Calculate correlation matrix
                try:
                    # Create a copy of the dataframe with only the essential metrics
                    essential_df = df[valid_metrics].copy()
                    
                    # Drop any rows with nulls
                    essential_df = essential_df.dropna()
                    
                    # Calculate correlation
                    corr_matrix = essential_df.corr(method='pearson')
                    
                    # Format readable metric names
                    readable_corr_metrics = {}
                    for metric in valid_metrics:
                        if metric.startswith("framework_accuracy_"):
                            readable_corr_metrics[metric] = "Acc: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                        elif metric == "framework_agreement":
                            readable_corr_metrics[metric] = "Framework Agmt"
                        elif metric == "semantic_similarity":
                            readable_corr_metrics[metric] = "Semantic Sim"
                        elif metric == "reasoning_count":
                            readable_corr_metrics[metric] = "Reasoning Count"
                        elif metric == "avg_framework_confidence":
                            readable_corr_metrics[metric] = "Avg Framework Conf"
                        else:
                            readable_corr_metrics[metric] = metric.replace("_", " ").title()
                    
                    # Rename for display
                    corr_matrix.columns = [readable_corr_metrics[m] for m in corr_matrix.columns]
                    corr_matrix.index = [readable_corr_metrics[m] for m in corr_matrix.index]
                    
                    # Convert to markdown
                    md_file.write("### Correlation Matrix Between Key Metrics\n\n")
                    md_file.write(corr_matrix.round(2).to_markdown() + "\n\n")
                    
                    md_file.write("*Note: Correlation values range from -1 (perfect negative correlation) to 1 (perfect positive correlation). 0 indicates no correlation.*\n\n")
                except Exception as e:
                    logger.warning(f"Error generating correlation matrix: {e}")
            
            # Statistical significance analysis
            md_file.write("## Statistical Significance Analysis\n\n")
            
            # Define key metrics for statistical testing
            statistical_metrics = []

            # Add framework_agreement if available
            if "framework_agreement" in df.columns and len(df['framework_agreement'].dropna()) >= 10:
                statistical_metrics.append("framework_agreement")

            # Add semantic_similarity if available  
            if "semantic_similarity" in df.columns and len(df['semantic_similarity'].dropna()) >= 10:
                statistical_metrics.append("semantic_similarity")

            # Add accuracy metrics if available
            for col in df.columns:
                if col.startswith("framework_accuracy_") and len(df[col].dropna()) >= 10:
                    statistical_metrics.append(col)

            # Check if we have any metrics for analysis
            if not statistical_metrics:
                md_file.write("*Insufficient data for statistical analysis. Statistical tests require at least 10 data points per metric.*\n\n")
            else:
                try:
                    # Try to import scipy for statistical tests
                    from scipy import stats
                    import numpy as np
                    
                    for metric in statistical_metrics:
                        # Format metric name for display
                        if metric.startswith("framework_accuracy_"):
                            display_metric = "Accuracy: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                        elif metric == "framework_agreement":
                            display_metric = "Framework Agreement"
                        elif metric == "semantic_similarity":
                            display_metric = "Semantic Similarity"
                        else:
                            display_metric = metric.replace("_", " ").title()
                            
                        md_file.write(f"### {display_metric}\n\n")
                        
                        # Check if we have enough models for ANOVA
                        if len(df['model'].unique()) >= 2:
                            # Check for evidence of repeated runs
                            test_case_model_pairs = df.groupby(['test_case', 'model']).size().reset_index(name='count') 
                            has_repeated_measures = test_case_model_pairs['count'].max() > 1

                            if has_repeated_measures:
                                # First aggregate to respect repeated measures design (proper approach)
                                md_file.write("*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*\n\n")
                                
                                # Aggregate by test_case and model first to avoid pseudoreplication
                                aggregated_data = df.groupby(['test_case', 'model'])[metric].mean().reset_index()
                                # Get models for analysis
                                models = aggregated_data['model'].unique()
                                # Get the test cases for analysis
                                test_cases_list = aggregated_data['test_case'].unique()
                                
                                # Then create groups by MODEL (not by case as the variable name suggested)
                                groups = [aggregated_data[aggregated_data['model'] == model][metric].dropna() for model in models]
                            else:
                                # Original approach for non-repeated data
                                models = df['model'].unique()
                                test_cases_list = df['test_case'].unique()
                                groups = [df[df['model'] == model][metric].dropna() for model in models]
                            
                            # Filter out empty groups
                            groups = [group for group in groups if len(group) > 1]
                            
                            if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                                try:
                                    # Traditional one-way ANOVA
                                    # f_statistic, p_value = stats.f_oneway(*groups)
                                    
                                    # Instead, let's implement Repeated Measures ANOVA
                                    from statsmodels.stats.anova import AnovaRM
                                    
                                    # Prepare data for RM ANOVA
                                    rm_data = []
                                    
                                    # Use the original dataframe to preserve test case information
                                    for test_case in df['test_case'].unique():
                                        for model in df['model'].unique():
                                            case_model_data = df[(df['test_case'] == test_case) & (df['model'] == model)]
                                            if len(case_model_data) > 0:
                                                # Use mean if multiple repetitions
                                                value = case_model_data[metric].mean()
                                                rm_data.append({
                                                    'test_case': test_case,
                                                    'model': model,
                                                    'value': value
                                                })
                                    
                                    rm_df = pd.DataFrame(rm_data)
                                    
                                    # Check if we have enough data for RM ANOVA
                                    if len(rm_df) > 0 and len(rm_df['test_case'].unique()) >= 2:
                                        # Run RM ANOVA
                                        rm_anova = AnovaRM(rm_df, 'value', 'test_case', within=['model'])
                                        rm_anova_result = rm_anova.fit()
                                        
                                        # Extract results
                                        result_df = rm_anova_result.anova_table
                                        f_value = result_df.loc['model', 'F Value']
                                        p_value = result_df.loc['model', 'Pr > F']
                                        num_df = result_df.loc['model', 'Num DF']
                                        den_df = result_df.loc['model', 'Den DF']
                                        
                                        md_file.write("#### Repeated Measures ANOVA\n\n")
                                        md_file.write(f"Repeated Measures ANOVA test for differences in {display_metric.lower()} between models (accounting for test case variability):\n\n")
                                        md_file.write(f"- F-value: {f_value:.4f}\n")
                                        md_file.write(f"- p-value: {p_value:.4f}\n")
                                        md_file.write(f"- Degrees of freedom: {num_df:.0f}, {den_df:.0f}\n\n")
                                        
                                        if p_value < 0.05:
                                            md_file.write("The p-value is less than 0.05, suggesting there is a **statistically significant difference** in ")
                                            md_file.write(f"{display_metric.lower()} between models when controlling for test case variability.\n\n")
                                            md_file.write("This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, ")
                                            md_file.write("which provides increased statistical power to detect true differences between models.\n\n")
                                        else:
                                            md_file.write("The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a ")
                                            md_file.write(f"statistically significant difference in {display_metric.lower()} between models when controlling for test case variability.\n\n")
                                            md_file.write("This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, ")
                                            md_file.write("which provides a more accurate assessment than treating each test case as independent.\n\n")
                                    else:
                                        # Fall back to one-way ANOVA if we don't have enough data for RM ANOVA
                                        f_statistic, p_value = stats.f_oneway(*groups)
                                        
                                        md_file.write("#### One-way ANOVA (Fallback)\n\n")
                                        md_file.write(f"One-way ANOVA test for differences in {display_metric.lower()} between models (insufficient data for Repeated Measures ANOVA):\n\n")
                                        md_file.write(f"- F-statistic: {f_statistic:.4f}\n")
                                        md_file.write(f"- p-value: {p_value:.4f}\n\n")
                                        
                                        if p_value < 0.05:
                                            md_file.write("The p-value is less than 0.05, suggesting there is a **statistically significant difference** in ")
                                            md_file.write(f"{display_metric.lower()} between at least some of the models.\n\n")
                                        else:
                                            md_file.write("The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a ")
                                            md_file.write(f"statistically significant difference in {display_metric.lower()} between the models.\n\n")
                                    
                                    # Store p-value for use in post-hoc tests
                                    p_value = p_value  # This ensures p_value is defined for post-hoc tests regardless of which method we used
                                        
                                    # If significant, add Tukey's HSD test for post-hoc analysis
                                    if p_value < 0.05 and len(groups) >= 3:
                                        md_file.write("#### Post-hoc Analysis (Tukey's HSD Test)\n\n")
                                        
                                        try:
                                            # Create a DataFrame in the format required for Tukey's test
                                            tukey_data = []
                                            for i, model in enumerate(df['model'].unique()):
                                                model_data = df[df['model'] == model][metric].dropna()
                                                if len(model_data) > 0:
                                                    for value in model_data:
                                                        tukey_data.append({'model': model, 'value': value})
                                            
                                            tukey_df = pd.DataFrame(tukey_data)
                                            
                                            if len(tukey_df) > 0:
                                                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                                tukey_result = pairwise_tukeyhsd(tukey_df['value'], tukey_df['model'], alpha=0.05)
                                                
                                                # Convert to markdown table
                                                md_file.write("| Group 1 | Group 2 | Mean Diff | p-value | Significant |\n")
                                                md_file.write("|---------|---------|-----------|---------|-------------|\n")
                                                
                                                for i in range(len(tukey_result.groupsunique)):
                                                    for j in range(i+1, len(tukey_result.groupsunique)):
                                                        idx = (i * (len(tukey_result.groupsunique) - 1) - i*(i-1)//2 + j - i - 1) 
                                                        if idx < len(tukey_result.pvalues):
                                                            group1 = tukey_result.groupsunique[i]
                                                            group2 = tukey_result.groupsunique[j]
                                                            mean_diff = tukey_result.meandiffs[idx]
                                                            p_value = tukey_result.pvalues[idx]
                                                            is_significant = "Yes" if p_value < 0.05 else "No"
                                                            
                                                            md_file.write(f"| {group1} | {group2} | {mean_diff:.4f} | {p_value:.4f} | {is_significant} |\n")
                                        except Exception as e:
                                            md_file.write(f"*Note: Could not perform Tukey's HSD test: {str(e)}*\n\n")
                                except Exception as e:
                                    md_file.write(f"*Error performing ANOVA: {str(e)}*\n\n")
                            else:
                                md_file.write("*Not enough groups with sufficient data to perform ANOVA.*\n\n")
                                
                            # Descriptive statistics by model
                            md_file.write("#### Descriptive Statistics by Model\n\n")
                            
                            # Calculate mean, std, min, max, and 95% CI for each model
                            stats_data = []
                            for model in df['model'].unique():
                                model_data = df[df['model'] == model][metric].dropna()
                                if len(model_data) >= 2:  # Need at least 2 data points for std and CI
                                    mean = model_data.mean()
                                    std = model_data.std()
                                    min_val = model_data.min()
                                    max_val = model_data.max()
                                    count = len(model_data)
                                    
                                    # Calculate 95% confidence interval
                                    sem = stats.sem(model_data)
                                    ci_95 = stats.t.interval(0.95, count-1, loc=mean, scale=sem)
                                    
                                    stats_data.append({
                                        'Model': model,
                                        'Mean': mean,
                                        'Std': std,
                                        'Min': min_val,
                                        'Max': max_val,
                                        'Count': count,
                                        'CI_Lower': ci_95[0],
                                        'CI_Upper': ci_95[1]
                                    })
                            
                            if stats_data:
                                # Create markdown table
                                md_file.write("| Model | Mean | Std | Min | Max | Count | 95% CI |\n")
                                md_file.write("|-------|------|-----|-----|-----|-------|---------|\n")
                                
                                # Sort by mean, descending
                                for row in sorted(stats_data, key=lambda x: x['Mean'], reverse=True):
                                    md_file.write(f"| {row['Model']} | {row['Mean']:.3f} | {row['Std']:.3f} | ")
                                    md_file.write(f"{row['Min']:.3f} | {row['Max']:.3f} | {row['Count']} | ")
                                    md_file.write(f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}] |\n")
                                
                                md_file.write("\n")
                            else:
                                md_file.write("*Insufficient data for descriptive statistics.*\n\n")
                                
                            # Effect Size Analysis (Cohen's d)
                            if len(df['model'].unique()) >= 2:
                                md_file.write("#### Effect Size Analysis (Cohen's d)\n\n")
                                md_file.write("Cohen's d measures the standardized difference between two means. It indicates the magnitude of the effect:\n\n")
                                md_file.write("- Small effect: d < 0.5\n")
                                md_file.write("- Medium effect: 0.5 ≤ d < 0.8\n")
                                md_file.write("- Large effect: d ≥ 0.8\n\n")
                                
                                # Add note about correction for multiple comparisons
                                models = list(df['model'].unique())
                                num_comparisons = len(models) * (len(models) - 1) // 2
                                md_file.write(f"**Note:** Benjamini-Hochberg False Discovery Rate (FDR) correction has been applied to control for {num_comparisons} multiple comparisons.\n")
                                md_file.write("When multiple statistical tests are performed, the probability of observing a significant result by chance increases.\n")
                                md_file.write("The Benjamini-Hochberg procedure adjusts p-values to control the expected proportion of false discoveries among all rejected hypotheses.\n")
                                md_file.write("Effect sizes labeled with '(corrected)' indicate comparisons that were not statistically significant after correction.\n\n")
                                
                                # Create markdown table
                                md_file.write("| Model 1 | Model 2 | Cohen's d | Effect Size |\n")
                                md_file.write("|---------|---------|-----------|-------------|\n")
                                
                                has_valid_comparison = False
                                
                                # First pass: collect all cohen's d values and calculate p-values
                                pairwise_data = []
                                
                                for i in range(len(models)):
                                    for j in range(i+1, len(models)):
                                        model1 = models[i]
                                        model2 = models[j]
                                        
                                        data1 = df[df['model'] == model1][metric].dropna()
                                        data2 = df[df['model'] == model2][metric].dropna()
                                        
                                        if len(data1) >= 2 and len(data2) >= 2:
                                            try:
                                                # Cohen's d calculation
                                                mean1, mean2 = data1.mean(), data2.mean()
                                                n1, n2 = len(data1), len(data2)
                                                
                                                # Pooled standard deviation
                                                s1, s2 = data1.std(), data2.std()
                                                
                                                if s1 == 0 and s2 == 0:
                                                    # Both distributions are constant
                                                    cohen_d = 0.0
                                                    raw_effect_size = "None (constant values)"
                                                    p_value = 1.0  # No significant difference
                                                else:
                                                    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                                                    
                                                    # Avoid division by zero
                                                    if pooled_std == 0:
                                                        cohen_d = 0.0
                                                        raw_effect_size = "None (no variation)"
                                                        p_value = 1.0  # No significant difference
                                                    else:
                                                        cohen_d = abs(mean1 - mean2) / pooled_std
                                                        
                                                        # Calculate p-value from t-test for use in Benjamini-Hochberg FDR correction
                                                        try:
                                                            from scipy import stats
                                                            _, p_value = stats.ttest_ind(data1, data2, equal_var=True)
                                                        except:
                                                            # If t-test fails, use a conservative approximation based on Cohen's d
                                                            # This is a rough approximation for when the test isn't available
                                                            p_value = 2 * (1 - stats.norm.cdf(abs(cohen_d) / 2))
                                                        
                                                        # Store raw effect size for later adjustment
                                                        if cohen_d < 0.5:
                                                            raw_effect_size = "Small"
                                                        elif cohen_d < 0.8:
                                                            raw_effect_size = "Medium"
                                                        else:
                                                            raw_effect_size = "Large"
                                                
                                                # Store comparison data for later application of Benjamini-Hochberg FDR
                                                pairwise_data.append({
                                                    'model1': model1,
                                                    'model2': model2,
                                                    'cohen_d': cohen_d,
                                                    'raw_effect_size': raw_effect_size,
                                                    'p_value': p_value
                                                })
                                                has_valid_comparison = True
                                            except Exception as e:
                                                md_file.write(f"| {model1} | {model2} | Error | {str(e)} |\n")
                                
                                # Apply Benjamini-Hochberg FDR correction if we have valid comparisons
                                if has_valid_comparison and len(pairwise_data) > 0:
                                    # Sort by p-value (lowest to highest)
                                    pairwise_data.sort(key=lambda x: x['p_value'])
                                    
                                    # Apply Benjamini-Hochberg FDR correction
                                    alpha = 0.05  # Traditional significance level
                                    n = len(pairwise_data)
                                    
                                    # Calculate adjusted p-values using Benjamini-Hochberg procedure
                                    for i, comparison in enumerate(pairwise_data):
                                        # BH step: calculate critical value based on rank
                                        bh_critical = alpha * (i + 1) / n
                                        # Store the critical value
                                        comparison['bh_critical'] = bh_critical
                                        # Check significance against BH critical value
                                        comparison['is_significant'] = comparison['p_value'] <= bh_critical
                                    
                                    # Adjust effect size interpretation based on corrected significance
                                    for comparison in pairwise_data:
                                        if not comparison['is_significant']:
                                            # If not significant after correction, downgrade effect size interpretation
                                            if comparison['raw_effect_size'] == "Medium":
                                                comparison['adjusted_effect_size'] = "Small (corrected)"
                                            elif comparison['raw_effect_size'] == "Large":
                                                comparison['adjusted_effect_size'] = "Medium (corrected)"
                                            else:
                                                comparison['adjusted_effect_size'] = comparison['raw_effect_size']
                                        else:
                                            # Keep original effect size if still significant
                                            comparison['adjusted_effect_size'] = comparison['raw_effect_size']
                                    
                                    # Write corrected results to markdown
                                    for comparison in pairwise_data:
                                        md_file.write(f"| {comparison['model1']} | {comparison['model2']} | {comparison['cohen_d']:.3f} | {comparison['adjusted_effect_size']} |\n")
                                
                                if not has_valid_comparison:
                                    md_file.write("*No valid comparisons could be made for effect size calculation.*\n")
                                
                                md_file.write("\n")
                except ImportError:
                    md_file.write("*Note: Statistical significance analysis requires the scipy and statsmodels packages, which were not available.*\n\n")
                except Exception as e:
                    md_file.write(f"*Error in statistical analysis: {str(e)}*\n\n")

            # Per-Case Analysis
            md_file.write("## Per-Case Analysis\n\n")
            md_file.write("This section analyzes performance across different test cases to identify which cases are particularly challenging or easy for language models.\n\n")
            
            if 'test_case' in df.columns and len(df['test_case'].unique()) > 1:
                test_cases = df['test_case'].unique()
                md_file.write(f"The evaluation dataset contains {len(test_cases)} distinct test cases.\n\n")
                
                # Get all framework accuracy metrics
                case_metrics = []
                for col in df.columns:
                    if col.startswith("framework_accuracy_") and not df[col].isna().all():
                        case_metrics.append(col)
                
                # Add standard metrics if available
                if "framework_agreement" in df.columns and not df["framework_agreement"].isna().all():
                    case_metrics.append("framework_agreement")
                if "semantic_similarity" in df.columns and not df["semantic_similarity"].isna().all():
                    case_metrics.append("semantic_similarity")
                
                if not case_metrics:
                    md_file.write("*No valid metrics available for per-case analysis.*\n\n")
                else:
                    # Calculate aggregate statistics by test case
                    md_file.write("### Overall Case Difficulty\n\n")
                    md_file.write("This table shows the average performance across all models for each test case, sorted from highest to lowest performance.\n\n")
                    
                    # Create a case summary dataframe
                    case_summaries = []
                    
                    for test_case in test_cases:
                        case_data = df[df['test_case'] == test_case]
                        case_summary = {'Test Case': test_case}
                        
                        # Calculate average metrics for this case
                        for metric in case_metrics:
                            if not case_data[metric].isna().all():
                                metric_values = case_data[metric].dropna()
                                if len(metric_values) > 0:
                                    case_summary[metric] = metric_values.mean()
                        
                        # Only add if we have at least one valid metric
                        if len(case_summary) > 1:  # More than just the Test Case column
                            case_summaries.append(case_summary)
                    
                    if case_summaries:
                        # Convert to DataFrame
                        case_df = pd.DataFrame(case_summaries)
                        
                        # Calculate average score across all metrics for sorting
                        metric_cols = [col for col in case_df.columns if col != 'Test Case']
                        if metric_cols:
                            case_df['Average Score'] = case_df[metric_cols].mean(axis=1)
                            case_df = case_df.sort_values('Average Score', ascending=False)
                            
                            # Format metrics for display
                            md_file.write("| Test Case | " + " | ".join([m.replace("framework_accuracy_", "Acc: ").replace("framework_agreement", "Framework Agmt").replace("semantic_similarity", "Semantic Sim").replace("_", " ").title() for m in metric_cols]) + " | Average |\n")
                            md_file.write("|" + "-" * 11 + "|" + "".join(["-" * 10 + "|" for _ in metric_cols]) + "-" * 10 + "|\n")
                            
                            for _, row in case_df.iterrows():
                                test_case = row['Test Case']
                                metrics_values = [f"{row[m]:.3f}" for m in metric_cols]
                                avg_score = row['Average Score']
                                md_file.write(f"| {test_case} | " + " | ".join(metrics_values) + f" | {avg_score:.3f} |\n")
                                
                            md_file.write("\n")
                            
                            # Identify easiest and most challenging cases
                            if len(case_df) >= 2:
                                top_n = min(3, len(case_df))
                                bottom_n = min(3, len(case_df))
                                
                                md_file.write("#### Easiest Test Cases\n\n")
                                md_file.write("The following test cases had the highest average performance across all models and metrics:\n\n")
                                for i, (_, row) in enumerate(case_df.head(top_n).iterrows()):
                                    md_file.write(f"{i+1}. **{row['Test Case']}** - Average Score: {row['Average Score']:.3f}\n")
                                md_file.write("\n")
                                
                                md_file.write("#### Most Challenging Test Cases\n\n")
                                md_file.write("The following test cases had the lowest average performance across all models and metrics:\n\n")
                                for i, (_, row) in enumerate(case_df.tail(bottom_n).iterrows()):
                                    md_file.write(f"{i+1}. **{row['Test Case']}** - Average Score: {row['Average Score']:.3f}\n")
                                md_file.write("\n")
                    else:
                        md_file.write("*No valid summary data available for test cases.*\n\n")
                
                    # Detailed framework-specific case analysis
                    frameworks = [col.replace("framework_accuracy_", "") for col in df.columns if col.startswith("framework_accuracy_")]
                    
                    for framework in frameworks:
                        accuracy_col = f"framework_accuracy_{framework}"
                        match_col = f"match_{framework}"
                        gold_col = f"gold_{framework}"
                        
                        if all(col in df.columns for col in [accuracy_col, match_col, gold_col]):
                            md_file.write(f"### {framework.replace('_', ' ').title()} Framework Case Analysis\n\n")
                            
                            # Filter out rows where gold standard is null
                            framework_df = df[~df[gold_col].isin(["null", "None", "", "Null"])]
                            framework_df = framework_df[~framework_df[gold_col].isna()]
                            
                            if not framework_df.empty:
                                # Calculate success rates by test case
                                case_success = framework_df.groupby('test_case')[match_col].mean().reset_index()
                                case_success = case_success.sort_values(match_col, ascending=False)
                                
                                # Add gold standard values
                                case_gold = framework_df.groupby('test_case')[gold_col].first().reset_index()
                                case_success = case_success.merge(case_gold, on='test_case')
                                
                                # Calculate model count and correct model count
                                case_counts = []
                                for test_case in case_success['test_case']:
                                    case_data = framework_df[framework_df['test_case'] == test_case]
                                    total_models = len(case_data['model'].unique())
                                    correct_models = len(case_data[case_data[match_col] > 0]['model'].unique())
                                    case_counts.append({
                                        'test_case': test_case,
                                        'total_models': total_models,
                                        'correct_models': correct_models
                                    })
                                
                                counts_df = pd.DataFrame(case_counts)
                                case_success = case_success.merge(counts_df, on='test_case')
                                
                                # Format for display
                                md_file.write("| Test Case | Gold Standard | Success Rate | Correct Models | Total Models |\n")
                                md_file.write("|-----------|---------------|-------------|----------------|-------------|\n")
                                
                                for _, row in case_success.iterrows():
                                    md_file.write(f"| {row['test_case']} | {row[gold_col]} | {row[match_col]:.3f} | ")
                                    md_file.write(f"{int(row['correct_models'])} | {int(row['total_models'])} |\n")
                                
                                md_file.write("\n")
                            else:
                                md_file.write(f"*No valid gold standard data available for {framework} framework.*\n\n")
                    
                    # Statistical analysis of per-case variance
                    try:
                        md_file.write("### Statistical Comparison of Case Difficulty\n\n")
                        
                        from scipy import stats
                        
                        # Check if we have enough cases for analysis
                        if len(test_cases) >= 5 and len(case_metrics) >= 1:
                            # Use one consistent metric for analysis
                            analysis_metric = case_metrics[0]  # Use first available metric
                            
                            # Format metric name for display
                            if analysis_metric.startswith("framework_accuracy_"):
                                display_metric = "Accuracy: " + analysis_metric.replace("framework_accuracy_", "").replace("_", " ").title()
                            elif analysis_metric == "framework_agreement":
                                display_metric = "Framework Agreement"
                            elif analysis_metric == "semantic_similarity": 
                                display_metric = "Semantic Similarity"
                            else:
                                display_metric = analysis_metric.replace("_", " ").title()
                                
                            md_file.write(f"We analyzed whether there are statistically significant differences in {display_metric} across test cases.\n\n")
                            
                            # Perform one-way ANOVA to test if differences between cases are significant
                            # Check for evidence of repeated runs
                            test_case_model_pairs = df.groupby(['test_case', 'model']).size().reset_index(name='count') 
                            has_repeated_measures = test_case_model_pairs['count'].max() > 1
                            
                            if has_repeated_measures:
                                # First aggregate to respect repeated measures design (proper approach)
                                md_file.write("*Note: Data contains repeated measurements of the same case-model combinations. Using aggregated means for ANOVA to maintain statistical validity.*\n\n")
                                
                                # Aggregate by test_case and model first to avoid pseudoreplication
                                aggregated_data = df.groupby(['test_case', 'model'])[analysis_metric].mean().reset_index()
                                # Get the list of test cases for analysis
                                test_cases_list = aggregated_data['test_case'].unique()
                                
                                # Then perform ANOVA comparing CASES (not models in this section)
                                case_groups = [aggregated_data[aggregated_data['test_case'] == case][analysis_metric].dropna() for case in test_cases_list]
                            else:
                                # Original approach for non-repeated data
                                test_cases_list = df['test_case'].unique()
                                case_groups = [df[df['test_case'] == case][analysis_metric].dropna() for case in test_cases_list]
                            
                            # Filter out empty groups
                            case_groups = [group for group in case_groups if len(group) > 1]
                            
                            if len(case_groups) >= 2:  # Need at least 2 groups for ANOVA
                                try:
                                    # Traditional one-way ANOVA
                                    # f_statistic, p_value = stats.f_oneway(*case_groups)
                                    
                                    # Instead, let's implement Repeated Measures ANOVA
                                    from statsmodels.stats.anova import AnovaRM
                                    
                                    # Prepare data for RM ANOVA
                                    rm_data = []
                                    
                                    # Use the original dataframe to preserve test case information
                                    for test_case in df['test_case'].unique():
                                        for model in df['model'].unique():
                                            case_model_data = df[(df['test_case'] == test_case) & (df['model'] == model)]
                                            if len(case_model_data) > 0:
                                                # Use mean if multiple repetitions
                                                value = case_model_data[analysis_metric].mean()
                                                rm_data.append({
                                                    'test_case': test_case,
                                                    'model': model,
                                                    'value': value
                                                })
                                    
                                    rm_df = pd.DataFrame(rm_data)
                                    
                                    # Check if we have enough data for RM ANOVA
                                    if len(rm_df) > 0 and len(rm_df['test_case'].unique()) >= 2:
                                        # Run RM ANOVA
                                        rm_anova = AnovaRM(rm_df, 'value', 'test_case', within=['model'])
                                        rm_anova_result = rm_anova.fit()
                                        
                                        # Extract results
                                        result_df = rm_anova_result.anova_table
                                        f_value = result_df.loc['model', 'F Value']
                                        p_value = result_df.loc['model', 'Pr > F']
                                        num_df = result_df.loc['model', 'Num DF']
                                        den_df = result_df.loc['model', 'Den DF']
                                        
                                        md_file.write("#### Repeated Measures ANOVA\n\n")
                                        md_file.write(f"Repeated Measures ANOVA test for differences in {display_metric.lower()} between models (accounting for test case variability):\n\n")
                                        md_file.write(f"- F-value: {f_value:.4f}\n")
                                        md_file.write(f"- p-value: {p_value:.4f}\n")
                                        md_file.write(f"- Degrees of freedom: {num_df:.0f}, {den_df:.0f}\n\n")
                                        
                                        if p_value < 0.05:
                                            md_file.write("The p-value is less than 0.05, suggesting there is a **statistically significant difference** in ")
                                            md_file.write(f"{display_metric.lower()} between models when controlling for test case variability.\n\n")
                                            md_file.write("This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, ")
                                            md_file.write("which provides increased statistical power to detect true differences between models.\n\n")
                                        else:
                                            md_file.write("The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a ")
                                            md_file.write(f"statistically significant difference in {display_metric.lower()} between models when controlling for test case variability.\n\n")
                                            md_file.write("This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, ")
                                            md_file.write("which provides a more accurate assessment than treating each test case as independent.\n\n")
                                    else:
                                        # Fall back to one-way ANOVA if we don't have enough data for RM ANOVA
                                        f_statistic, p_value = stats.f_oneway(*case_groups)
                                        
                                        md_file.write("#### One-way ANOVA (Fallback)\n\n")
                                        md_file.write(f"One-way ANOVA test for differences in {display_metric.lower()} between models (insufficient data for Repeated Measures ANOVA):\n\n")
                                        md_file.write(f"- F-statistic: {f_statistic:.4f}\n")
                                        md_file.write(f"- p-value: {p_value:.4f}\n\n")
                                        
                                        if p_value < 0.05:
                                            md_file.write("The p-value is less than 0.05, suggesting there is a **statistically significant difference** in ")
                                            md_file.write(f"{display_metric.lower()} between at least some of the models.\n\n")
                                        else:
                                            md_file.write("The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a ")
                                            md_file.write(f"statistically significant difference in {display_metric.lower()} between the models.\n\n")
                                    
                                    # Store p-value for use in post-hoc tests
                                    p_value = p_value  # This ensures p_value is defined for post-hoc tests regardless of which method we used
                                        
                                    # If significant, add Tukey's HSD test for post-hoc analysis
                                    if p_value < 0.05 and len(case_groups) >= 3:
                                        md_file.write("#### Post-hoc Analysis (Tukey's HSD Test)\n\n")
                                        
                                        try:
                                            # Create a DataFrame in the format required for Tukey's test
                                            tukey_data = []
                                            for i, model in enumerate(df['model'].unique()):
                                                model_data = df[df['model'] == model][analysis_metric].dropna()
                                                if len(model_data) > 0:
                                                    for value in model_data:
                                                        tukey_data.append({'model': model, 'value': value})
                                            
                                            tukey_df = pd.DataFrame(tukey_data)
                                            
                                            if len(tukey_df) > 0:
                                                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                                tukey_result = pairwise_tukeyhsd(tukey_df['value'], tukey_df['model'], alpha=0.05)
                                                
                                                # Convert to markdown table
                                                md_file.write("| Group 1 | Group 2 | Mean Diff | p-value | Significant |\n")
                                                md_file.write("|---------|---------|-----------|---------|-------------|\n")
                                                
                                                for i in range(len(tukey_result.groupsunique)):
                                                    for j in range(i+1, len(tukey_result.groupsunique)):
                                                        idx = (i * (len(tukey_result.groupsunique) - 1) - i*(i-1)//2 + j - i - 1) 
                                                        if idx < len(tukey_result.pvalues):
                                                            group1 = tukey_result.groupsunique[i]
                                                            group2 = tukey_result.groupsunique[j]
                                                            mean_diff = tukey_result.meandiffs[idx]
                                                            p_value = tukey_result.pvalues[idx]
                                                            is_significant = "Yes" if p_value < 0.05 else "No"
                                                            
                                                            md_file.write(f"| {group1} | {group2} | {mean_diff:.4f} | {p_value:.4f} | {is_significant} |\n")
                                        except Exception as e:
                                            md_file.write(f"*Note: Could not perform Tukey's HSD test: {str(e)}*\n\n")
                                except Exception as e:
                                    md_file.write(f"*Error performing ANOVA: {str(e)}*\n\n")
                            else:
                                md_file.write("*Not enough test cases with sufficient data to perform statistical analysis.*\n\n")
                        else:
                            md_file.write("*Insufficient data for statistical comparison of test cases.*\n\n")
                    except ImportError:
                        md_file.write("*Note: Statistical comparison requires the scipy package, which was not available.*\n\n")
                    except Exception as e:
                        md_file.write(f"*Error in case statistical analysis: {str(e)}*\n\n")
                    
                    # Model consistency across cases
                    md_file.write("### Model Consistency Across Test Cases\n\n")
                    md_file.write("This analysis examines whether models perform consistently across different test cases or if performance varies significantly by case.\n\n")
                    
                    if len(test_cases) >= 3 and len(df['model'].unique()) >= 2:
                        # Choose the most common metric for analysis
                        analysis_metric = None
                        max_count = 0
                        
                        for metric in case_metrics:
                            count = len(df[~df[metric].isna()])
                            if count > max_count:
                                max_count = count
                                analysis_metric = metric
                        
                        if analysis_metric:
                            # Format metric name for display
                            if analysis_metric.startswith("framework_accuracy_"):
                                display_metric = "Accuracy: " + analysis_metric.replace("framework_accuracy_", "").replace("_", " ").title()
                            elif analysis_metric == "framework_agreement":
                                display_metric = "Framework Agreement"
                            elif analysis_metric == "semantic_similarity":
                                display_metric = "Semantic Similarity"
                            else:
                                display_metric = analysis_metric.replace("_", " ").title()
                            
                            # Calculate coefficient of variation (CV) for each model across cases
                            model_consistency = []
                            
                            for model in df['model'].unique():
                                model_data = df[df['model'] == model]
                                
                                if len(model_data) >= 3:  # Need at least 3 data points for meaningful CV
                                    # Calculate mean and standard deviation across test cases
                                    case_means = []
                                    for case in test_cases:
                                        case_data = model_data[model_data['test_case'] == case][analysis_metric].dropna()
                                        if len(case_data) > 0:
                                            case_means.append(case_data.mean())
                                    
                                    if len(case_means) >= 3:
                                        mean = np.mean(case_means)
                                        std = np.std(case_means)
                                        
                                        # Coefficient of variation (CV) - lower means more consistent
                                        if mean > 0:
                                            cv = std / mean
                                            model_consistency.append({
                                                'Model': model,
                                                'Mean': mean,
                                                'Std': std,
                                                'CV': cv,
                                                'Case Count': len(case_means)
                                            })
                            
                            if model_consistency:
                                # Sort by CV (ascending - more consistent first)
                                model_consistency_df = pd.DataFrame(model_consistency)
                                model_consistency_df = model_consistency_df.sort_values('CV')
                                
                                md_file.write(f"#### Model Consistency in {display_metric}\n\n")
                                md_file.write("The coefficient of variation (CV) measures how consistent a model's performance is across different test cases.\n")
                                md_file.write("Lower CV values indicate more consistent performance across test cases.\n\n")
                                
                                md_file.write("| Model | Mean | Std Dev | Coefficient of Variation | # Cases |\n")
                                md_file.write("|-------|------|---------|--------------------------|--------|\n")
                                
                                for _, row in model_consistency_df.iterrows():
                                    md_file.write(f"| {row['Model']} | {row['Mean']:.3f} | {row['Std']:.3f} | {row['CV']:.3f} | {int(row['Case Count'])} |\n")
                                
                                md_file.write("\n")
                                
                                # Add interpretation
                                most_consistent = model_consistency_df.iloc[0]['Model']
                                least_consistent = model_consistency_df.iloc[-1]['Model']
                                
                                md_file.write(f"**{most_consistent}** shows the most consistent performance across different test cases ")
                                md_file.write(f"(CV: {model_consistency_df.iloc[0]['CV']:.3f}), while **{least_consistent}** shows the ")
                                md_file.write(f"most variable performance (CV: {model_consistency_df.iloc[-1]['CV']:.3f}).\n\n")
                            else:
                                md_file.write("*Insufficient data to calculate model consistency across test cases.*\n\n")
                        else:
                            md_file.write("*No suitable metric found for consistency analysis.*\n\n")
                    else:
                        md_file.write("*Insufficient data for consistency analysis (need at least 3 test cases and 2 models).*\n\n")
            else:
                md_file.write("*Test case information not available for analysis.*\n\n")
            
            # Reliability Analysis (for repeated runs)
            md_file.write("## Reliability Analysis\n\n")
            md_file.write("To ensure scientific rigor, each case was tested multiple times with each model. This section analyzes the consistency of model performance across repeated runs of the same test cases.\n\n")
            
            if 'test_case' in df.columns and 'model' in df.columns:
                # Check if we have evidence of repeated runs in the data
                test_case_model_pairs = df.groupby(['test_case', 'model']).size().reset_index(name='count')
                
                if test_case_model_pairs['count'].max() > 1:
                    # We have repeated runs
                    repeat_count = test_case_model_pairs['count'].max()
                    md_file.write(f"Each case-model combination was tested up to {repeat_count} times to assess reliability.\n\n")
                    
                    # Select metrics for consistency analysis
                    reliability_metrics = []
                    
                    for col in df.columns:
                        if (col.startswith("framework_accuracy_") or 
                            col == "framework_agreement" or 
                            col == "semantic_similarity") and not df[col].isna().all():
                            reliability_metrics.append(col)
                    
                    if not reliability_metrics:
                        md_file.write("*No suitable metrics found for reliability analysis.*\n\n")
                    else:
                        md_file.write("### Within-Model Reliability\n\n")
                        md_file.write("This table shows consistency metrics for each model across repeated runs of the same test cases.\n\n")
                        
                        # Calculate reliability metrics
                        model_reliability = []
                        
                        for model in df['model'].unique():
                            model_data = df[df['model'] == model]
                            model_info = {'Model': model}
                            
                            # For each metric, calculate ICC and within-subject CV
                            for metric in reliability_metrics:
                                # Get data for this metric, removing NaNs
                                metric_data = model_data[['test_case', metric]].dropna()
                                
                                # Need at least some repetitions to calculate reliability
                                if len(metric_data) > len(metric_data['test_case'].unique()):
                                    try:
                                        # Calculate ICC (Intraclass Correlation Coefficient)
                                        # Prep data in format for ICC calculation
                                        case_groups = []
                                        for case in metric_data['test_case'].unique():
                                            case_values = metric_data[metric_data['test_case'] == case][metric].values
                                            if len(case_values) > 1:  # Need at least 2 measurements
                                                case_groups.append(case_values)
                                        
                                        if len(case_groups) >= 2:
                                            # Calculate Within-Subject CV (Coefficient of Variation)
                                            # This measures consistency across repeated runs
                                            within_subject_cvs = []
                                            
                                            for case_values in case_groups:
                                                if len(case_values) > 1 and np.mean(case_values) > 0:
                                                    cv = np.std(case_values) / np.mean(case_values)
                                                    within_subject_cvs.append(cv)
                                            
                                            if within_subject_cvs:
                                                mean_cv = np.mean(within_subject_cvs)
                                                model_info[f"{metric}_CV"] = mean_cv
                                                
                                                # Add interpretation of reliability
                                                if mean_cv < 0.1:
                                                    model_info[f"{metric}_Reliability"] = "Excellent"
                                                elif mean_cv < 0.2:
                                                    model_info[f"{metric}_Reliability"] = "Good"
                                                elif mean_cv < 0.3:
                                                    model_info[f"{metric}_Reliability"] = "Moderate"
                                                else:
                                                    model_info[f"{metric}_Reliability"] = "Poor"
                                                    
                                                # Also calculate absolute agreement percentage for framework matches
                                                if metric.startswith("framework_accuracy_") or metric == "framework_agreement":
                                                    perfect_matches = 0
                                                    total_cases = 0
                                                    
                                                    for case in metric_data['test_case'].unique():
                                                        case_results = metric_data[metric_data['test_case'] == case][metric].values
                                                        if len(case_results) > 1:
                                                            # Check if all values are identical
                                                            if np.all(case_results == case_results[0]):
                                                                perfect_matches += 1
                                                            total_cases += 1
                                                    
                                                    if total_cases > 0:
                                                        agreement_pct = perfect_matches / total_cases * 100
                                                        model_info[f"{metric}_Agreement"] = agreement_pct
                                    except Exception as e:
                                        logger.warning(f"Error calculating reliability for {model}, {metric}: {e}")
                            
                            # Only add to results if we have at least one reliability metric
                            if len(model_info) > 1:
                                model_reliability.append(model_info)
                        
                        if model_reliability:
                            # Create DataFrame for display
                            reliability_df = pd.DataFrame(model_reliability)
                            
                            # Format column names for readability
                            display_cols = {}
                            for col in reliability_df.columns:
                                if col == "Model":
                                    display_cols[col] = "Model"
                                elif col.endswith("_CV"):
                                    metric = col.replace("_CV", "")
                                    if metric.startswith("framework_accuracy_"):
                                        display = "CV: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                                    elif metric == "framework_agreement":
                                        display = "CV: Framework Agreement"
                                    elif metric == "semantic_similarity":
                                        display = "CV: Semantic Similarity"
                                    else:
                                        display = "CV: " + metric.replace("_", " ").title()
                                    display_cols[col] = display
                                elif col.endswith("_Reliability"):
                                    metric = col.replace("_Reliability", "")
                                    if metric.startswith("framework_accuracy_"):
                                        display = "Reliability: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                                    elif metric == "framework_agreement":
                                        display = "Reliability: Framework Agreement"
                                    elif metric == "semantic_similarity":
                                        display = "Reliability: Semantic Similarity"
                                    else:
                                        display = "Reliability: " + metric.replace("_", " ").title()
                                    display_cols[col] = display
                                elif col.endswith("_Agreement"):
                                    metric = col.replace("_Agreement", "")
                                    if metric.startswith("framework_accuracy_"):
                                        display = "Agreement %: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                                    elif metric == "framework_agreement":
                                        display = "Agreement %: Framework Agreement"
                                    else:
                                        display = "Agreement %: " + metric.replace("_", " ").title()
                                    display_cols[col] = display
                            
                            # Sort models by average CV (ascending = more consistent first)
                            cv_cols = [col for col in reliability_df.columns if col.endswith("_CV")]
                            if cv_cols:
                                reliability_df['Avg_CV'] = reliability_df[cv_cols].mean(axis=1)
                                reliability_df = reliability_df.sort_values('Avg_CV')
                            
                            # For each metric, create a section in the markdown report
                            for metric in reliability_metrics:
                                cv_col = f"{metric}_CV"
                                reliability_col = f"{metric}_Reliability"
                                agreement_col = f"{metric}_Agreement"
                                
                                if cv_col in reliability_df.columns:
                                    # Format metric name for display
                                    if metric.startswith("framework_accuracy_"):
                                        display_metric = "Accuracy: " + metric.replace("framework_accuracy_", "").replace("_", " ").title()
                                    elif metric == "framework_agreement":
                                        display_metric = "Framework Agreement"
                                    elif metric == "semantic_similarity":
                                        display_metric = "Semantic Similarity"
                                    else:
                                        display_metric = metric.replace("_", " ").title()
                                        
                                    md_file.write(f"#### Reliability for {display_metric}\n\n")
                                    
                                    # Create a table showing reliability metrics
                                    headers = ["Model", "CV (lower is better)"]
                                    if reliability_col in reliability_df.columns:
                                        headers.append("Reliability Rating")
                                    if agreement_col in reliability_df.columns:
                                        headers.append("% Identical Results")
                                    
                                    md_file.write("| " + " | ".join(headers) + " |\n")
                                    md_file.write("|" + "--|".join(["--" for _ in headers]) + "|\n")
                                    
                                    # Add rows for each model
                                    for _, row in reliability_df.iterrows():
                                        model_row = [row['Model'], f"{row[cv_col]:.3f}"]
                                        if reliability_col in reliability_df.columns:
                                            model_row.append(str(row[reliability_col]))  # Ensure string conversion
                                        if agreement_col in reliability_df.columns:
                                            model_row.append(f"{row[agreement_col]:.1f}%")
                                        
                                        md_file.write("| " + " | ".join(model_row) + " |\n")
                                    
                                    md_file.write("\n")
                                    
                                    # Add interpretation
                                    if reliability_df[cv_col].notna().any():
                                        most_reliable = reliability_df.iloc[0]['Model']
                                        least_reliable = reliability_df.iloc[-1]['Model']
                                        
                                        md_file.write(f"**{most_reliable}** shows the most consistent results across repeated runs ")
                                        md_file.write(f"(CV: {reliability_df.iloc[0][cv_col]:.3f}), while **{least_reliable}** shows the ")
                                        md_file.write(f"most variable results (CV: {reliability_df.iloc[-1][cv_col]:.3f}).\n\n")
                                        
                                        if agreement_col in reliability_df.columns:
                                            # Find models with perfect agreement (100%)
                                            perfect_models = reliability_df[reliability_df[agreement_col] == 100]['Model'].tolist()
                                            if perfect_models:
                                                if len(perfect_models) == 1:
                                                    md_file.write(f"**{perfect_models[0]}** achieved perfect consistency, producing identical results across all repeated runs.\n\n")
                                                else:
                                                    md_file.write(f"The following models achieved perfect consistency, producing identical results across all repeated runs: ")
                                                    md_file.write(", ".join([f"**{model}**" for model in perfect_models]) + ".\n\n")
                            
                            # Add explanation of CV interpretation
                            md_file.write("### Interpretation of Reliability Metrics\n\n")
                            md_file.write("The Coefficient of Variation (CV) measures the consistency of results across repeated runs of the same test cases:\n\n")
                            md_file.write("- **Excellent reliability**: CV < 0.1 (less than 10% variation)\n")
                            md_file.write("- **Good reliability**: CV < 0.2 (less than 20% variation)\n")
                            md_file.write("- **Moderate reliability**: CV < 0.3 (less than 30% variation)\n")
                            md_file.write("- **Poor reliability**: CV ≥ 0.3 (30% or greater variation)\n\n")
                            
                            md_file.write("The **% Identical Results** shows how often a model produced exactly the same result on all runs of the same test case.\n\n")
                            
                            # Scientific implications
                            md_file.write("### Scientific Implications\n\n")
                            md_file.write("Reliability analysis is critical for scientific rigor when evaluating language models. High reliability indicates:\n\n")
                            md_file.write("1. **Deterministic behavior**: Models producing identical results on repeated runs demonstrate deterministic outputs.\n")
                            md_file.write("2. **Performance stability**: Low variation across runs suggests reliable performance metrics that can be trusted.\n")
                            md_file.write("3. **Scientific validity**: Models with higher reliability produce results that are more scientifically valid and reproducible.\n\n")
                            
                            md_file.write("When interpreting performance metrics in this report, consider each model's reliability alongside its raw performance. ")
                            md_file.write("Models with inconsistent results (high CV) may show inflated performance in some metrics due to chance rather than true capability.\n\n")
                        else:
                            md_file.write("*Insufficient data to calculate reliability metrics.*\n\n")
                else:
                    md_file.write("*No evidence of repeated runs found in the dataset. Each case-model combination appears to have been tested only once.*\n\n")
            else:
                md_file.write("*Test case or model information not available for reliability analysis.*\n\n")
                
            # Update Statistical Significance Analysis section to account for repeated measures
            # This would go in the ANOVA test section of the existing code
            try:
                md_file.write("### Repeated Measures Analysis\n\n")
                md_file.write("Since each case was tested multiple times with each model, we conducted a Repeated Measures ANOVA to account for this experimental design.\n\n")
                md_file.write("We analyzed " + display_metric + " using our Repeated Measures ANOVA approach to properly account for the correlation structure in repeated measurements.\n\n")
                md_file.write("See the 'Statistical Methodology' section for details about how Repeated Measures ANOVA provides a more rigorous analysis than traditional ANOVA for this experimental design.\n\n")
                
                # Remove the old one-way ANOVA on case means section and replace with RM ANOVA explanation
                md_file.write("#### Statistical Methodology\n\n")
                md_file.write("**Note on statistical methodology**: This analysis employs Repeated Measures ANOVA, which directly models the within-subject factor (models) ")
                md_file.write("and the between-subject factor (test cases). Unlike traditional one-way ANOVA, this approach properly accounts for the correlation structure in ")
                md_file.write("repeated measurements, providing greater statistical power and more accurate p-values by controlling for the natural variability between test cases.\n\n")
                md_file.write("This method avoids pseudoreplication and inflated degrees of freedom that would occur if treating each repetition as an independent observation, ")
                md_file.write("resulting in a more rigorous and scientifically sound analysis than traditional ANOVA approaches.\n\n")
                
                if 'test_case' in df.columns and 'model' in df.columns:
                    # Check if we have evidence of repeated runs in the data
                    test_case_model_pairs = df.groupby(['test_case', 'model']).size().reset_index(name='count')
                    
                    if test_case_model_pairs['count'].max() > 1 and len(valid_metrics) > 0:
                        # Choose a key metric for analysis
                        primary_metric = None
                        
                        # Prioritize framework_agreement and accuracy metrics
                        if "framework_agreement" in valid_metrics:
                            primary_metric = "framework_agreement"
                        elif any(m.startswith("framework_accuracy_") for m in valid_metrics):
                            primary_metric = [m for m in valid_metrics if m.startswith("framework_accuracy_")][0]
                        else:
                            primary_metric = valid_metrics[0]
                        
                        # Format metric name for display
                        if primary_metric.startswith("framework_accuracy_"):
                            display_metric = "Accuracy: " + primary_metric.replace("framework_accuracy_", "").replace("_", " ").title()
                        elif primary_metric == "framework_agreement":
                            display_metric = "Framework Agreement"
                        elif primary_metric == "semantic_similarity":
                            display_metric = "Semantic Similarity"
                        else:
                            display_metric = primary_metric.replace("_", " ").title()
                            
                        md_file.write(f"We analyzed {display_metric} using an approach that accounts for repeated measurements.\n\n")
                        
                        try:
                            # For scientific rigor, calculate means per test_case/model combination
                            # This is a proper approach for analyzing repeated measures
                            aggregated_data = df.groupby(['test_case', 'model'])[primary_metric].mean().reset_index()
                            
                            # Now perform ANOVA on the aggregated data (one observation per test_case/model)
                            models = aggregated_data['model'].unique()
                            
                            if len(models) >= 2:
                                # Create groups for ANOVA
                                groups = [aggregated_data[aggregated_data['model'] == model][primary_metric].values for model in models]
                                groups = [group for group in groups if len(group) > 1]  # Need at least 2 values per group
                                
                                if len(groups) >= 2:
                                    # We've replaced this with Repeated Measures ANOVA
                                    # For reference, leaving the old code commented out
                                    """
                                    f_statistic, p_value = stats.f_oneway(*groups)
                                    
                                    md_file.write("#### One-way ANOVA on Case Means\n\n")
                                    md_file.write(f"One-way ANOVA test for differences in {display_metric.lower()} between models (using case means):\n\n")
                                    md_file.write(f"- F-statistic: {f_statistic:.4f}\n")
                                    md_file.write(f"- p-value: {p_value:.4f}\n\n")
                                    
                                    if p_value < 0.05:
                                        md_file.write("The p-value is less than 0.05, suggesting there is a **statistically significant difference** in ")
                                        md_file.write(f"{display_metric.lower()} between at least some of the models, even when accounting for repeated measures.\n\n")
                                    else:
                                        md_file.write("The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a ")
                                        md_file.write(f"statistically significant difference in {display_metric.lower()} between the models when accounting for repeated measures.\n\n")
                                    """
                                    
                                    # Now we'll implement RM ANOVA here since we have the data already
                                    try:
                                        # Prepare data for RM ANOVA
                                        rm_data = []
                                        for test_case in df['test_case'].unique():
                                            for model in df['model'].unique():
                                                case_model_data = df[(df['test_case'] == test_case) & (df['model'] == model)]
                                                if len(case_model_data) > 0:
                                                    # Use mean if multiple repetitions
                                                    value = case_model_data[primary_metric].mean()
                                                    rm_data.append({
                                                        'test_case': test_case,
                                                        'model': model,
                                                        'value': value
                                                    })
                                        
                                        rm_df = pd.DataFrame(rm_data)
                                        
                                        if len(rm_df) > 0 and len(rm_df['test_case'].unique()) >= 2:
                                            # Import needed for RM ANOVA
                                            from statsmodels.stats.anova import AnovaRM
                                            
                                            # Run RM ANOVA
                                            rm_anova = AnovaRM(rm_df, 'value', 'test_case', within=['model'])
                                            rm_anova_result = rm_anova.fit()
                                            
                                            # Extract results
                                            result_df = rm_anova_result.anova_table
                                            f_value = result_df.loc['model', 'F Value']
                                            p_value = result_df.loc['model', 'Pr > F']
                                            num_df = result_df.loc['model', 'Num DF']
                                            den_df = result_df.loc['model', 'Den DF']
                                            
                                            md_file.write("#### Repeated Measures ANOVA\n\n")
                                            md_file.write(f"Repeated Measures ANOVA test for differences in {display_metric.lower()} between models (accounting for test case variability):\n\n")
                                            md_file.write(f"- F-value: {f_value:.4f}\n")
                                            md_file.write(f"- p-value: {p_value:.4f}\n")
                                            md_file.write(f"- Degrees of freedom: {num_df:.0f}, {den_df:.0f}\n\n")
                                            
                                            if p_value < 0.05:
                                                md_file.write("The p-value is less than 0.05, suggesting there is a **statistically significant difference** in ")
                                                md_file.write(f"{display_metric.lower()} between models when controlling for test case variability.\n\n")
                                                md_file.write("This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, ")
                                                md_file.write("which provides increased statistical power to detect true differences between models.\n\n")
                                            else:
                                                md_file.write("The p-value is greater than 0.05, suggesting there is not enough evidence to conclude a ")
                                                md_file.write(f"statistically significant difference in {display_metric.lower()} between models when controlling for test case variability.\n\n")
                                                md_file.write("This analysis properly accounts for the repeated measures design by modeling test cases as a random factor, ")
                                                md_file.write("which provides a more accurate assessment than treating each test case as independent.\n\n")
                                        else:
                                            md_file.write("*Not enough valid data across test cases to perform Repeated Measures ANOVA.*\n\n")
                                    
                                    except Exception as e:
                                        md_file.write(f"*Error performing Repeated Measures ANOVA: {str(e)}*\n\n")
                                else:
                                    md_file.write("*Not enough valid data across models to perform repeated measures ANOVA.*\n\n")
                            else:
                                md_file.write("*Insufficient number of models for repeated measures ANOVA.*\n\n")
                        except Exception as e:
                            md_file.write(f"*Error performing repeated measures analysis: {str(e)}*\n\n")
                    else:
                        md_file.write("*No evidence of repeated measures design in the dataset.*\n\n")
                else:
                    md_file.write("*Test case or model information not available for repeated measures analysis.*\n\n")
            except Exception as e:
                md_file.write(f"*Error in repeated measures analysis: {str(e)}*\n\n")
            
            # Conclusion
            md_file.write("## Conclusion\n\n")
            
            # Calculate overall best model based on average performance across all valid metrics
            if valid_metrics and 'model' in df.columns:
                model_averages = df.groupby('model')[valid_metrics].mean().mean(axis=1).sort_values(ascending=False)
                
                if not model_averages.empty:
                    best_model = model_averages.index[0]
                    best_score = model_averages.iloc[0]
                    
                    md_file.write(f"Based on the comprehensive analysis of all metrics, **{best_model}** demonstrates the strongest overall performance ")
                    md_file.write(f"with an average score of {best_score:.3f} across all evaluation criteria.\n\n")
            
            md_file.write("This report provides a comprehensive overview of the performance of various language models ")
            md_file.write("on forensic analysis tasks. The data can be used to draw conclusions about the relative strengths ")
            md_file.write("and weaknesses of different models in understanding and applying forensic frameworks.\n\n")
            
            md_file.write("*End of Report*\n")
        
        logger.info(f"Generated markdown report at {md_file_path}")
        
    except Exception as e:
        logger.error(f"Error generating markdown report: {e}")
        import traceback
        logger.error(traceback.format_exc())

def create_visualizations(results: List[Dict], plots_dir: Path) -> None:
    """Main function to create all desired visualizations.
    
    Args:
        results: List of evaluation result dictionaries (already loaded)
        plots_dir: Directory to save plots
    """
    # Create the horizontal grouped bar chart
    create_horizontal_grouped_bar_chart(results, plots_dir)
    
    # Generate the markdown report for LLM consumption
    generate_markdown_report(results, plots_dir)
    
    # Can add more visualization functions here as needed 