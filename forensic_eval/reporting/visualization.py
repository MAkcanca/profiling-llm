"""Visualization utilities for evaluation results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from cycler import cycler

# Set up logger
logger = logging.getLogger(__name__)

# Publication-ready settings - Professional for scientific papers
SMALL_SIZE = 9
MEDIUM_SIZE = 11
LARGE_SIZE = 14
TITLE_SIZE = 16

# Use Nature/Science-style formatting for plots
plt.style.use(['seaborn-v0_8-whitegrid', 'seaborn-v0_8-paper'])

# Figure settings for publication-quality
mpl.rcParams['figure.figsize'] = (10, 6)  # Default figure size
mpl.rcParams['figure.dpi'] = 300  # High resolution for publication
mpl.rcParams['savefig.dpi'] = 300  # High resolution for saving
mpl.rcParams['savefig.bbox'] = 'tight'  # Tight bounding box when saving

# Academic font settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = SMALL_SIZE
mpl.rcParams['axes.titlesize'] = LARGE_SIZE
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = MEDIUM_SIZE
mpl.rcParams['axes.labelweight'] = 'regular'
mpl.rcParams['xtick.labelsize'] = SMALL_SIZE
mpl.rcParams['ytick.labelsize'] = SMALL_SIZE
mpl.rcParams['legend.fontsize'] = SMALL_SIZE
mpl.rcParams['figure.titlesize'] = TITLE_SIZE
mpl.rcParams['figure.titleweight'] = 'bold'

# Professional axis appearance
mpl.rcParams['axes.linewidth'] = 0.8  # Thinner axis lines for professional look
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.spines.top'] = False  # Remove top spine
mpl.rcParams['axes.spines.right'] = False  # Remove right spine
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = ':'  # Dotted grid lines
mpl.rcParams['grid.linewidth'] = 0.6
mpl.rcParams['grid.alpha'] = 0.4

# Improved tick parameters
mpl.rcParams['xtick.major.size'] = 3.5
mpl.rcParams['ytick.major.size'] = 3.5
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['xtick.minor.visible'] = False
mpl.rcParams['ytick.minor.visible'] = False

# Publication-quality color palettes
# Nature/Science style color palettes - colorblind-friendly
PAPER_PALETTE = sns.color_palette([
    "#4878D0", "#EE854A", "#6ACC64", "#D65F5F",
    "#956CB4", "#8C613C", "#DC7EC0", "#797979",
    "#82C6E2", "#D5BB67"
])

# Academic blues palette for sequential data
BLUES_PALETTE = sns.color_palette("Blues_r", 8)

# Professional gradient for bar charts
BAR_PALETTE = sns.color_palette([
    "#08519c", "#3182bd", "#6baed6", "#9ecae1",
    "#c6dbef", "#deebf7", "#fee0d2", "#fcbba1",
    "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d"
])

# Set default color cycle
mpl.rcParams['axes.prop_cycle'] = cycler('color', PAPER_PALETTE)

# Default seaborn settings
sns.set_context("paper")
sns.set_style("whitegrid", {
    'axes.edgecolor': '.8',
    'grid.color': '.8',
    'grid.linestyle': ':',
})

# Apply consistent error bar styling
mpl.rcParams['errorbar.capsize'] = 3

def generate_model_comparison_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate plots comparing different models.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
    """
    try:
        # Skip if no model column or it's empty
        if 'model' not in df.columns or df['model'].isna().all():
            logger.warning("No model data available for comparison plots")
            return
        
        # Essential metrics for model comparison
        metrics = ["reasoning_count", "avg_framework_confidence"]
        
        # Filter metrics that exist in the dataframe
        valid_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
        
        if not valid_metrics:
            logger.warning("No valid metrics for model comparison plots")
            return
            
        plt.figure(figsize=(8, 6))
        
        for i, metric in enumerate(valid_metrics):
            plt.subplot(len(valid_metrics), 1, i+1)
            # Handle error bars safely
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax = sns.barplot(x="model", y=metric, data=df, errorbar=("ci", 95), palette=PAPER_PALETTE)
                
                # Add value labels above bars with enough padding
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.2f}", 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', fontsize=SMALL_SIZE,
                              xytext=(0, 5), textcoords='offset points')
                
            plt.title(f"Model Comparison: {metric.replace('_', ' ').title()}", fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout(pad=1.5)  # Add padding to prevent overlap
        
        plt.savefig(plots_dir / "model_comparison.png", bbox_inches='tight')
        plt.close()
        
        # Generate gold standard accuracy plot
        generate_gold_standard_accuracy_plot(df, plots_dir)
    except Exception as e:
        logger.error(f"Error generating model comparison plots: {e}")

def generate_gold_standard_accuracy_plot(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate publication-quality accuracy plots for gold standard comparison.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
    """
    try:
        # Define the essential accuracy metrics
        accuracy_metrics = [
            "framework_agreement",     # How well the model matches gold standard framework
            "semantic_similarity"      # Semantic similarity to gold standard
        ]
        
        # Add any framework accuracy columns if they exist
        for col in df.columns:
            if col.startswith("framework_accuracy_") and not df[col].isna().all():
                accuracy_metrics.append(col)
        
        # Filter to metrics that actually exist in the data
        valid_metrics = [m for m in accuracy_metrics if m in df.columns and not df[m].isna().all()]
        
        if not valid_metrics:
            logger.warning("No accuracy metrics found for gold standard comparison")
            return
        
        # Create individual plots for each metric with publication quality
        for metric in valid_metrics:
            # Format the title to be more readable
            title = metric.replace("framework_accuracy_", "Accuracy: ")
            title = title.replace("framework_agreement", "Framework Agreement")
            title = title.replace("semantic_similarity", "Semantic Similarity")
            title = title.replace("_", " ").title()
            
            # Create a copy of the dataframe for this specific metric
            # Filter out rows where this metric is null (indicating framework is not applicable)
            metric_df = df.copy()
            metric_df = metric_df[~metric_df[metric].isna()]
            
            # Skip if filtering results in empty dataframe
            if metric_df.empty:
                logger.info(f"No data for metric {metric} after filtering null values")
                continue
            
            # Create figure with appropriate dimensions - HORIZONTAL orientation
            plt.figure(figsize=(11, max(6, 0.4 * len(metric_df['model'].unique()))))
            
            # Reset to base style for clean academic look
            with plt.style.context('seaborn-v0_8-whitegrid'):
                # Sort models by performance for better visualization
                # Use descending instead of ascending to stack from highest to lowest
                sorted_df = metric_df.sort_values(by=metric, ascending=False)  # Descending for horizontal bars - highest on top
                
                # Calculate 95% confidence intervals to determine label positioning
                model_groups = sorted_df.groupby('model')
                ci_high_values = {}
                
                for name, group in model_groups:
                    # Calculate the 95% CI upper bound
                    values = group[metric].dropna()
                    if len(values) > 1:
                        ci = sns.utils.ci(values, which=95)
                        mean = values.mean()
                        ci_high = mean + (ci[1] - mean)  # Upper bound of CI
                        ci_high_values[name] = ci_high
                    else:
                        ci_high_values[name] = values.iloc[0] if not values.empty else 0
                
                # Find maximum CI value for x-axis limit
                max_ci_value = max(ci_high_values.values()) if ci_high_values else 1.0
                
                # Create horizontal bar chart
                aggregated = metric_df.groupby("model", as_index=False)[metric].mean().sort_values(by=metric, ascending=False)
                order = aggregated["model"].tolist()

                # Create the horizontal bar plot without the hue parameter
                ax = sns.barplot(
                    x=metric,
                    y="model",
                    data=aggregated,
                    order=order,
                    errorbar=None,  # or errorbar=("ci", 95) if you want confidence intervals on the aggregated data
                    palette="Blues_d",
                    orient='h'
                )
                
                # Add value labels positioned AFTER the error bars
                for i, p in enumerate(ax.patches):
                    model_name = p.get_y() + p.get_height()/2
                    width = p.get_width()
                    model = sorted_df.iloc[i]['model'] if i < len(sorted_df) else None
                    
                    if model in ci_high_values:
                        # Position label after the error bar
                        label_x = ci_high_values[model] + 0.02
                    else:
                        label_x = width + 0.05
                        
                    ax.annotate(
                        f"{width:.2f}",
                        (label_x, model_name),
                        ha='left', va='center',
                        fontsize=MEDIUM_SIZE-1,
                        fontweight='normal',
                        color='#444444'
                    )
                
                # Clean up the plot for academic publication
                plt.title(f"Gold Standard Comparison: {title}", fontweight="bold", fontsize=TITLE_SIZE, pad=20)
                
                # Improve x-axis
                plt.xlabel("Score", fontsize=MEDIUM_SIZE, fontweight="medium")
                
                # No y-axis label needed as model names are self-explanatory
                plt.ylabel("")
                
                # Set x-axis limits for accuracy metrics with space for labels
                if metric.startswith("framework_accuracy_") or metric == "framework_agreement" or metric == "semantic_similarity":
                    # Find minimum value with padding, but keep 0 as base if values are low
                    min_val = sorted_df[metric].min()
                    x_min = max(0, min_val - 0.05) if min_val > 0.2 else 0
                    
                    # Extend x-axis for labels that appear after CI error bars
                    x_max = min(1.2, max_ci_value + 0.25)  # Allow extra space for labels
                    plt.xlim(x_min, x_max)
                
                # Add subtle grid for readability (vertical only for horizontal bars)
                plt.grid(axis='x', linestyle='--', alpha=0.3)
                
                # Clean up spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.75)
                ax.spines['bottom'].set_linewidth(0.75)
                
                # Improve tick parameters
                plt.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, length=4, width=0.75)
                
                # Tighten layout
                plt.tight_layout(pad=2.0)
                
                # Create model abbreviations for legend if needed
                model_names = sorted_df["model"].unique()
                if any(len(name) > 20 for name in model_names) and len(model_names) > 5:
                    # Create a mapping dictionary
                    abbrev_dict = {}
                    for name in model_names:
                        if len(name) > 20:
                            parts = name.split('-')
                            if len(parts) > 1:
                                # Create abbreviation using first letter of each part
                                abbrev = ''.join(p[0].upper() for p in parts if p)
                                abbrev_dict[name] = abbrev
                    
                    # Only add legend if we created abbreviations
                    if abbrev_dict:
                        # Get current tick positions and labels
                        current_positions = ax.get_yticks()
                        current_labels = [label.get_text() for label in ax.get_yticklabels()]
                        
                        # Create new labels with abbreviations where needed
                        new_labels = [abbrev_dict.get(label, label) for label in current_labels]
                        
                        # Set the ticks and labels properly
                        ax.set_yticks(current_positions)
                        ax.set_yticklabels(new_labels)
                        
                        # Add a legend mapping with good formatting
                        legend_text = []
                        for orig, abbr in abbrev_dict.items():
                            legend_text.append(f'{abbr}: {orig}')
                        
                        # Place legend outside plot area
                        if legend_text:
                            plt.figtext(0.02, 0.02, '\n'.join(legend_text), 
                                     fontsize=SMALL_SIZE-1, 
                                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Save with publication quality
            clean_metric = metric.replace("framework_accuracy_", "accuracy_")
            plt.savefig(plots_dir / f"gold_standard_{clean_metric}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined visualization with all metrics in a single elegant figure
        if len(valid_metrics) > 1:
            # For combined view, use subplots in a column
            fig, axes = plt.subplots(
                nrows=len(valid_metrics), 
                ncols=1, 
                figsize=(11, 3.5 * len(valid_metrics)),
                constrained_layout=True  # Better than tight_layout for subplots
            )
            
            # Handle case of single subplot
            if len(valid_metrics) == 1:
                axes = [axes]
            
            # Process each metric with consistent formatting
            for i, metric in enumerate(valid_metrics):
                ax = axes[i]
                
                # Format title
                title = metric.replace("framework_accuracy_", "Accuracy: ")
                title = title.replace("framework_agreement", "Framework Agreement")
                title = title.replace("semantic_similarity", "Semantic Similarity")
                title = title.replace("_", " ").title()
                
                # Filter out rows where this metric is null (framework not applicable)
                metric_df = df.copy()
                metric_df = metric_df[~metric_df[metric].isna()]
                
                # Skip if no data
                if metric_df.empty:
                    ax.text(0.5, 0.5, f"No data for {title}", 
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=ax.transAxes)
                    continue
                
                # Sort models by performance
                # Use descending instead of ascending to stack from highest to lowest
                sorted_df = metric_df.sort_values(by=metric, ascending=False)  # Descending for horizontal bars - highest on top
                
                # Calculate CI for label positioning
                model_groups = sorted_df.groupby('model')
                ci_high_values = {}
                
                for name, group in model_groups:
                    # Calculate the 95% CI upper bound
                    values = group[metric].dropna()
                    if len(values) > 1:
                        ci = sns.utils.ci(values, which=95)
                        mean = values.mean()
                        ci_high = mean + (ci[1] - mean)  # Upper bound of CI
                        ci_high_values[name] = ci_high
                    else:
                        ci_high_values[name] = values.iloc[0] if not values.empty else 0
                
                # Find maximum CI value for x-axis limit
                max_ci_value = max(ci_high_values.values()) if ci_high_values else 1.0
                
                # Create horizontal bar plot
                sns.barplot(
                    x=metric,
                    y="model", 
                    data=sorted_df,
                    errorbar=("ci", 95),
                    ax=ax,
                    palette="Blues_d", 
                    orient='h',
                    hue="model",  # Set hue to model to avoid FutureWarning
                    legend=False  # Don't display redundant legend
                )
                
                # Add value labels positioned after error bars
                patch_indices = {}
                for j, p in enumerate(ax.patches):
                    model_name = sorted_df.iloc[j % len(sorted_df)]['model'] if j < len(sorted_df) * len(ax.patches) // len(sorted_df) else None
                    if model_name not in patch_indices:
                        patch_indices[model_name] = j
                
                for model_name, j in patch_indices.items():
                    if j < len(ax.patches):
                        p = ax.patches[j]
                        width = p.get_width()
                        model_y = p.get_y() + p.get_height()/2
                        
                        if model_name in ci_high_values:
                            # Position label after the error bar
                            label_x = ci_high_values[model_name] + 0.02
                        else:
                            label_x = width + 0.05
                            
                        ax.annotate(
                            f"{width:.2f}",
                            (label_x, model_y),
                            ha='left', va='center',
                            fontsize=SMALL_SIZE,
                            fontweight='normal',
                            color='#444444'
                        )
                
                # Clean up the plot
                ax.set_title(f"{title}", fontweight="bold", fontsize=LARGE_SIZE)
                ax.set_xlabel("Score" if i == len(valid_metrics)-1 else "")  # X-label only on bottom plot
                ax.set_ylabel("")  # No Y-label needed
                
                # Set x-axis limits with space for labels
                if metric.startswith("framework_accuracy_") or metric == "framework_agreement" or metric == "semantic_similarity":
                    min_val = sorted_df[metric].min()
                    x_min = max(0, min_val - 0.05) if min_val > 0.2 else 0
                    # Extended x-axis for error bars + labels
                    x_max = min(1.2, max_ci_value + 0.25)
                    ax.set_xlim(x_min, x_max)
                
                # Clean up spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.75)
                ax.spines['bottom'].set_linewidth(0.75)
                
                # Grid for readability
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                
                # Improve tick parameters
                ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, length=4, width=0.75)
            
            # Add a common title
            fig.suptitle("Gold Standard Comparison Metrics", 
                      fontsize=TITLE_SIZE, 
                      fontweight="bold", 
                      y=1.02)
            
            # Save the combined figure
            plt.savefig(plots_dir / "gold_standard_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # Generate a grouped bar chart comparing all metrics side by side
        generate_grouped_gold_standard_chart(df, plots_dir, valid_metrics)
            
    except Exception as e:
        logger.error(f"Error generating gold standard accuracy plot: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_grouped_gold_standard_chart(df: pd.DataFrame, plots_dir: Path, metrics: List[str]) -> None:
    """Generate a grouped (clustered) bar chart comparing all gold standard metrics.
    
    This allows for direct visual comparison of different metrics for each model.
    Creates publication-quality plots specifically optimized for scientific papers.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
        metrics: List of metrics to include in the chart
    """
    try:
        if not metrics or len(metrics) < 2:
            logger.info("Not enough metrics for grouped bar chart")
            return
            
        # Prepare data for grouped bar chart by melting the dataframe
        # First, create a copy with only necessary columns
        plot_df = df[['model'] + metrics].copy()
        
        # Filter out rows with NaN values in any of the metrics
        #plot_df = plot_df.dropna(subset=metrics)
        
        # If no data left, return
        if plot_df.empty:
            logger.warning("No data with complete metrics for grouped bar chart")
            return
            
        # Group by model and calculate the mean for each metric
        # This ensures we have one value per model per metric
        aggregated_df = plot_df.groupby('model')[metrics].mean().reset_index()
        
        # Sort models by the average of all metrics to get a sensible ordering
        aggregated_df['avg_score'] = aggregated_df[metrics].mean(axis=1)
        aggregated_df = aggregated_df.sort_values('avg_score', ascending=False)
        
        # Set up the order of models for consistent display
        model_order = aggregated_df['model'].tolist()
        
        # Melt the dataframe to get it in the right format for seaborn
        melted_df = pd.melt(
            aggregated_df, 
            id_vars=['model'], 
            value_vars=metrics,
            var_name='Metric', 
            value_name='Score'
        )
        
        # Format metric names for better readability
        melted_df['Metric'] = melted_df['Metric'].apply(lambda m: 
            m.replace("framework_accuracy_", "")  # Shorter labels
            .replace("framework_agreement", "Framework Agmt")
            .replace("semantic_similarity", "Semantic Sim")
            .replace("_", " ")
            .title()
        )

        # Map Metric names to abbreviations
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
        
        # Create abbreviated model names if needed for better readability in paper
        model_abbreviations = {
            "GPT-4.5-Preview": "GPT-4.5-Prev",
            "Claude-3.7-Sonnet-Thinking": "Claude-3.7-Sonnet-Think",
            "DeepSeek-R1": "DeepSeek-R1",
            "o3-mini": "o3-mini",
            "o3-mini-high": "o3-mini-high",
            "Llama-3.3-70B-Instruct": "Llama-3.3",
            "Gemini-2.0-Flash": "Gemini-2.0",
            "Gemini-2.0-Flash-Thinking-Exp0121": "Gemini-2.0-Think",
            "GPT-4o": "GPT-4o",
            "GPT-4o-mini": "GPT-4o-mini",
            "Claude-3.7-Sonnet": "Claude-3.7-Sonnet",
        }
        # Apply abbreviations if we have any
        if model_abbreviations:
            melted_df['display_model'] = melted_df['model'].apply(
                lambda m: model_abbreviations.get(m, m)
            )
        else:
            melted_df['display_model'] = melted_df['model']
        
        # Set up categorical order for abbreviated model names
        if model_abbreviations:
            model_display_order = [model_abbreviations.get(m, m) for m in model_order]
        else:
            model_display_order = model_order
        
        # Scientific paper-optimized figure size - compact but readable
        # Scale based on number of models and metrics, with constraints for readability
        n_models = len(model_order)
        n_metrics = len(metrics)
        fig_width = min(8, max(5, 3 + 0.5 * n_models))
        fig_height = min(6, max(3, 2 + 0.3 * n_models))

        # Publication-quality vertical bar chart - optimized for papers
        with plt.style.context(['seaborn-v0_8-whitegrid', 'seaborn-v0_8-paper']):
            # Use specific scientific color palette (colorblind-friendly)
            if n_metrics <= 4:
                # Scientific color scheme (Nature/Science-style) for up to 4 metrics
                palette = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F"][:n_metrics]
            else:
                # For many metrics, use a colorblind-friendly scientific palette
                palette = sns.color_palette("colorblind", n_metrics)
            
            # Create figure with precise dimensions for scientific papers
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
            
            # Create the vertical grouped bar chart
            sns.barplot(
                data=melted_df,
                x='display_model',  # Use abbreviated names if available
                y='Score',
                hue='Metric',
                palette=palette,
                ax=ax,
                width=0.8,  # Slightly narrower bars
                saturation=0.9  # Slightly more saturated for better visibility in paper
            )
            
            # Calculate bar width for proper positioning of text labels
            bar_width = 0.8 / n_metrics
            
            # Add value labels to each bar - optimized for compact paper format
            # Only add labels to bars above a certain threshold to avoid clutter
            threshold = 0.05
            
            # Find bars and organize by position
            bar_positions = {}
            for i, p in enumerate(ax.patches):
                # Calculate metric and model indices
                metric_idx = i % n_metrics
                model_idx = i // n_metrics
                
                # Add to dictionary if entry doesn't exist
                if model_idx not in bar_positions:
                    bar_positions[model_idx] = {}
                
                bar_positions[model_idx][metric_idx] = p
            
            # Now add labels for each model group
            for model_idx, metrics_dict in bar_positions.items():
                for metric_idx, p in metrics_dict.items():
                    height = p.get_height()
                    if height >= threshold:
                        # Position the label above the bar
                        ax.annotate(
                            f"{height:.2f}",
                            (p.get_x() + p.get_width()/2, height),
                            ha='center', va='bottom',
                            fontsize=SMALL_SIZE-1,  # Slightly smaller for paper
                            fontweight='normal',
                            color='#444444',
                            xytext=(0, 1),  # Reduced padding to save space
                            textcoords='offset points'
                        )
            
            # Scientific paper title and labels - concise but descriptive
            ax.set_title("Gold Standard Metric Comparison", fontweight="bold", fontsize=MEDIUM_SIZE, pad=10)
            ax.set_xlabel("")  # No x-label needed as model names are self-explanatory
            ax.set_ylabel("Score", fontsize=SMALL_SIZE+1, fontweight="medium")
            
            # Rotate x-axis labels for better fit in paper
            plt.xticks(rotation=45, ha="right", fontsize=SMALL_SIZE)
            plt.yticks(fontsize=SMALL_SIZE)
            
            # Set y-axis to start from 0 and extend to max value with minimal padding
            # This maximizes the data-to-ink ratio for scientific visualization
            max_val = melted_df['Score'].max()
            plt.ylim(0, min(1.05, max_val * 1.05))  # Reduced padding
            
            # Add subtle grid for readability - only horizontal lines for cleaner look
            plt.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.6)
            
            # Clean up spines for publication quality
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            ax.spines['left'].set_linewidth(0.6)
            ax.spines['bottom'].set_linewidth(0.6)
            
            # Add a compact legend suitable for scientific papers
            handles, labels = ax.get_legend_handles_labels()
            
            # Create a legend with optimal positioning for paper
            legend = plt.legend(
                handles, 
                labels, 
                title="Metric",
                loc='upper center',  # Center top position
                bbox_to_anchor=(0.5, -0.15),  # Position below the figure
                ncol=min(n_metrics, 3),  # Multiple columns for many metrics
                frameon=False,  # No frame for cleaner look
                fontsize=SMALL_SIZE-1,
                title_fontsize=SMALL_SIZE,
                columnspacing=1.0,  # Tighter spacing
                handletextpad=0.5  # Reduced space between color boxes and text
            )
            plt.setp(legend.get_title(), fontweight='bold')
            
            # Add model name mapping as a note if abbreviations were used
            if model_abbreviations:
                note_text = []
                for orig, abbr in model_abbreviations.items():
                    note_text.append(f"{abbr}: {orig}")
                
                # Add the mapping as a compact note
                if note_text:
                    plt.figtext(
                        0.02, 0.01, 
                        "Abbreviations: " + "; ".join(note_text),
                        fontsize=SMALL_SIZE-2, 
                        horizontalalignment='left',
                        style='italic'
                    )
            
            # Tight layout for paper-efficient use of space
            plt.tight_layout(rect=[0, 0.05, 1, 0.97])
            
            # Save the publication-ready figure
            plt.savefig(plots_dir / "gold_standard_grouped_comparison.png", dpi=400, bbox_inches='tight')
            plt.close()
            
            # Create horizontal version - often better for papers with many models
            # Scientific paper-optimized horizontal layout
            horizontal_fig_width = min(8, max(5, 3 + 0.3 * n_metrics))
            horizontal_fig_height = min(9, max(3, 1.5 + 0.5 * n_models))
            
            fig, ax = plt.subplots(figsize=(horizontal_fig_width, horizontal_fig_height), dpi=300)
            
            with plt.style.context(['seaborn-v0_8-whitegrid', 'seaborn-v0_8-paper']):
                # Create the horizontal bar plot optimized for paper publication
                sns.barplot(
                    data=melted_df,
                    x='Score',
                    y='display_model',  # Use abbreviated names if available
                    hue='Metric',
                    palette=palette,
                    orient='h',
                    ax=ax,
                    width=0.8,  # Slightly narrower bars for clarity
                    saturation=0.9  # Better visibility in print
                )
                
                # Add compact value labels - only for bars wide enough to read
                for p in ax.patches:
                    width = p.get_width()
                    # Only label if bar is long enough to fit text
                    if width >= 0.08:  # Higher threshold for horizontal layout
                        ax.annotate(
                            f"{width:.2f}",
                            (width, p.get_y() + p.get_height()/2),
                            ha='left', va='center',
                            fontsize=SMALL_SIZE-1,  # Slightly smaller for paper
                            fontweight='normal',
                            color='#444444',
                            xytext=(3, 0),  # Minimal offset
                            textcoords='offset points'
                        )
                
                # Scientific paper optimized title and labels
                ax.set_title("Gold Standard Metric Comparison", fontweight="bold", fontsize=MEDIUM_SIZE, pad=10)
                ax.set_xlabel("Score", fontsize=SMALL_SIZE+1, fontweight="medium")
                ax.set_ylabel("")  # No y-label needed
                
                # Set x-axis to start from 0 and extend to max value with minimal padding
                plt.xlim(0, min(1.05, max_val * 1.05))  # Reduced padding
                
                # Clean, subtle grid for enhanced readability
                plt.grid(axis='x', linestyle=':', alpha=0.3, linewidth=0.6)
                
                # Clean up spines for publication quality
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                ax.spines['left'].set_linewidth(0.6)
                ax.spines['bottom'].set_linewidth(0.6)
                
                # Optimize tick parameters for scientific publication
                plt.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, length=3, width=0.6)
                
                # Add legend suitable for scientific papers - positioned to minimize space usage
                legend = plt.legend(
                    loc='lower right',
                    frameon=False,  # No frame for cleaner look
                    fontsize=SMALL_SIZE-2,
                    title_fontsize=SMALL_SIZE,
                    handletextpad=0.5  # Reduced space between color boxes and text
                )
                plt.setp(legend.get_title(), fontweight='bold')
                
                # Tight layout optimized for paper
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                
                # Save the publication-ready horizontal figure
                plt.savefig(plots_dir / "gold_standard_grouped_comparison_horizontal.png", dpi=400, bbox_inches='tight')
                plt.close()
                
    except Exception as e:
        logger.error(f"Error generating grouped gold standard chart: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_framework_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate plots for framework-related metrics.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
    """
    try:
        # Skip if no model column
        if 'model' not in df.columns or df['model'].empty:
            return
            
        # Get framework confidence columns if they exist
        framework_cols = [col for col in df.columns if col.startswith("framework_confidence_") 
                         and not df[col].isna().all()]
        
        if not framework_cols:
            return
            
        plt.figure(figsize=(12, 8))
        
        try:
            # Melt the DataFrame to get a format suitable for seaborn
            melted_df = pd.melt(df, 
                              id_vars=["model", "test_case"], 
                              value_vars=framework_cols,
                              var_name="framework",
                              value_name="confidence")
            
            # Clean up framework names
            melted_df["framework"] = melted_df["framework"].apply(
                lambda x: x.replace("framework_confidence_", "").replace("_", " ").title())
            
            # Skip if melted dataframe is empty
            if melted_df.empty or melted_df["confidence"].isna().all():
                return
                
            # Create the plot
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax = sns.barplot(x="framework", y="confidence", hue="model", data=melted_df, 
                              errorbar=("ci", 95), palette=PAPER_PALETTE)
            plt.title("Framework Classification Confidence", fontweight="bold", pad=20)
            plt.xlabel("Framework")
            plt.ylabel("Confidence")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.ylim(0, 1.05)  # Confidence is between 0 and 1
            plt.tight_layout()
            
            plt.savefig(plots_dir / "framework_confidence.png", bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Error in framework barplot: {e}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error generating framework plots: {e}")

def generate_all_visualizations(results: List[Dict], plots_dir: Path) -> None:
    """Generate all visualizations from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        plots_dir: Directory to save plots
    """
    try:
        # Create plots directory
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Convert results to DataFrame safely
        try:
            # Flatten results for visualization
            flattened_results = []
            for result in results:
                # Skip invalid results
                if not isinstance(result, dict):
                    continue
                    
                # Get basic information
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
                    
                    # Add framework classifications
                    classifications = framework.get("classifications")
                    if classifications and isinstance(classifications, dict):
                        for fw_name, classification in classifications.items():
                            flat_result[f"classification_{fw_name}"] = str(classification)
                
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
                    
                    # Add framework match details
                    framework_matches = gold_standard.get("framework_matches")
                    if framework_matches and isinstance(framework_matches, dict):
                        for fw_name, match_info in framework_matches.items():
                            if isinstance(match_info, dict):
                                # Save match status (True/False)
                                flat_result[f"match_{fw_name}"] = match_info.get("match", False)
                                # Save the gold standard value
                                flat_result[f"gold_{fw_name}"] = str(match_info.get("gold", ""))
                
                flattened_results.append(flat_result)
            
            # Create DataFrame
            df = pd.DataFrame(flattened_results)
            
            # Skip if DataFrame is empty
            if df.empty:
                logger.warning("No valid data for visualization")
                return
                
            # Generate standard visualizations
            generate_model_comparison_plots(df, plots_dir)
            generate_framework_plots(df, plots_dir)
            
            # Explicitly generate gold standard accuracy plot
            generate_gold_standard_accuracy_plot(df, plots_dir)
            
            # Generate new scientific visualizations
            generate_confidence_interval_plots(df, plots_dir)
            generate_version_comparison_plots(df, plots_dir)
            generate_simplified_correlation_matrix(df, plots_dir)
            
            # Generate detailed model-case analysis for paper
            generate_detailed_prediction_analysis(df, plots_dir)
            
        except Exception as e:
            logger.error(f"Error creating DataFrame for visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def generate_confidence_interval_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate confidence interval plots for key metrics.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
    """
    try:
        # Skip if no model column
        if 'model' not in df.columns or df['model'].empty:
            return
            
        # Focus on essential metrics only
        metrics = [
            "reasoning_count", 
            "avg_framework_confidence", 
            "framework_agreement", 
            "semantic_similarity"
        ]
        
        # Add any framework accuracy columns if they exist
        for col in df.columns:
            if col.startswith("framework_accuracy_") and not df[col].isna().all():
                metrics.append(col)
        
        for metric in metrics:
            if metric in df.columns and not df[metric].isna().all():
                # Create a copy and filter out null values (indicating not applicable frameworks)
                plot_df = df.copy()
                plot_df = plot_df[~plot_df[metric].isna()]
                
                # Skip if filtering results in empty dataframe
                if plot_df.empty:
                    logger.info(f"No data for confidence interval plot of {metric} after filtering null values")
                    continue
                
                plt.figure(figsize=(10, 6))
                
                # Only include models with at least 2 data points for CI calculation
                model_counts = plot_df.groupby('model')[metric].count()
                valid_models = model_counts[model_counts >= 2].index.tolist()
                
                if not valid_models:
                    plt.close()
                    continue
                    
                plot_df = plot_df[plot_df['model'].isin(valid_models)]
                
                # Safety check - ensure we have data to plot
                if len(plot_df) == 0 or plot_df[metric].isna().all():
                    plt.close()
                    continue
                
                # Create the plot
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use linestyle='none' instead of join=False (deprecated)
                    # Use hue parameter to avoid FutureWarning
                    ax = sns.pointplot(
                        x="model", 
                        y=metric, 
                        data=plot_df,
                        linestyle='none',
                        errorbar=("ci", 95),
                        capsize=0.3,
                        color=PAPER_PALETTE[0],
                        hue="model",  # Set hue to model
                        legend=False  # Don't display redundant legend
                    )
                    
                    # Add value labels with better placement
                    # Safety check for valid models
                    for i, line in enumerate(ax.lines):
                        if i < len(valid_models):  # Safety check
                            x, y = line.get_xydata()[0]  # Get the x, y data from the line
                            model_name = valid_models[i]
                            mean_val = plot_df[plot_df['model'] == model_name][metric].mean()
                            ax.annotate(f"{mean_val:.2f}", 
                                     xy=(x, y), 
                                     xytext=(0, 15),  # Increased vertical distance
                                     textcoords='offset points',
                                     ha='center', va='bottom',
                                     fontsize=SMALL_SIZE)
                
                # Format the title to be more readable
                title = metric
                if metric.startswith("framework_accuracy_"):
                    title = "Accuracy: " + metric.replace("framework_accuracy_", "")
                elif metric == "framework_agreement":
                    title = "Framework Agreement"
                elif metric == "semantic_similarity":
                    title = "Semantic Similarity"
                
                title = title.replace("_", " ").title()
                
                plt.title(f"95% Confidence Intervals: {title}", fontweight="bold", pad=20)
                plt.xlabel("Model")
                plt.ylabel(title)
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Set y-axis from 0 to 1.0 for accuracy metrics (plus small margin for labels)
                if metric.startswith("framework_accuracy_") or metric == "framework_agreement" or metric == "semantic_similarity":
                    plt.ylim(0, min(1.05, max(plot_df[metric].max() * 1.1, 0.2)))
                    
                plt.tight_layout(pad=2.0)  # Add extra padding
                
                plt.savefig(plots_dir / f"{metric}_confidence_intervals.png", bbox_inches='tight')
                plt.close()
    except Exception as e:
        logger.error(f"Error generating confidence interval plots: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_version_comparison_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate plots comparing different versions of the same model.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
    """
    try:
        # Check if we have version information
        if 'version' not in df.columns or 'model' not in df.columns:
            logger.info("Missing version or model column for comparison plots")
            return
            
        if df['version'].nunique() <= 1:
            logger.info("No version data available for comparison plots")
            return
            
        # Focus on essential metrics only
        metrics = [
            "reasoning_count", 
            "avg_framework_confidence",
            "framework_agreement", 
            "semantic_similarity"
        ]
        
        # Add any framework accuracy columns if they exist
        for col in df.columns:
            if col.startswith("framework_accuracy_") and not df[col].isna().all():
                metrics.append(col)
        
        # Filter to metrics that exist in the dataframe
        valid_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
        
        if not valid_metrics:
            return
            
        # For each model, plot the metrics across versions
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Skip if only one version
            if model_df['version'].nunique() <= 1:
                continue
                
            # Get the number of valid metrics for this model
            model_metrics = []
            for metric in valid_metrics:
                # Only include metrics with non-null values
                metric_data = model_df[~model_df[metric].isna()]
                if not metric_data.empty and not metric_data[metric].isna().all():
                    model_metrics.append(metric)
                    
            if not model_metrics:
                continue
                
            # Calculate subplot layout
            n_metrics = len(model_metrics)
            n_cols = min(2, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
                
            plt.figure(figsize=(12, 8))
            
            for i, metric in enumerate(model_metrics):
                plt.subplot(n_rows, n_cols, i+1)
                
                # Filter out null values for this metric (framework not applicable)
                plot_df = model_df[~model_df[metric].isna()].copy()
                
                # Skip if filtering results in empty dataframe
                if plot_df.empty:
                    logger.info(f"No data for version comparison plot of {metric} for model {model} after filtering null values")
                    continue
                
                # Sort by version if numeric, otherwise use as is
                try:
                    if plot_df['version'].str.isnumeric().all():
                        plot_df = plot_df.sort_values('version', key=lambda x: x.astype(int))
                    else:
                        # Skip sort if not numeric
                        pass
                except:
                    # Skip sort on exception
                    pass
                    
                # Check if we have enough data to plot
                if len(plot_df) < 2 or plot_df[metric].isna().all():
                    continue
                        
                ax = sns.lineplot(x="version", y=metric, data=plot_df, marker='o', color=PAPER_PALETTE[1])
                
                # Add value annotations with better positioning
                for x, y in zip(plot_df['version'], plot_df[metric]):
                    ax.annotate(f"{y:.2f}", 
                              xy=(x, y), 
                              xytext=(0, 15),  # Increased vertical distance
                              textcoords='offset points',
                              ha='center', 
                              fontsize=SMALL_SIZE)
                
                # Format the title to be more readable
                title = metric
                if metric.startswith("framework_accuracy_"):
                    title = "Accuracy: " + metric.replace("framework_accuracy_", "")
                elif metric == "framework_agreement":
                    title = "Framework Agreement"
                elif metric == "semantic_similarity":
                    title = "Semantic Similarity"
                
                title = title.replace("_", " ").title()
                
                plt.title(title, fontweight="bold")
                plt.xlabel("Version")
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, linestyle='--', alpha=0.3)
                
                # Set y-axis from 0 to 1.0 for accuracy metrics
                if metric.startswith("framework_accuracy_") or metric == "framework_agreement" or metric == "semantic_similarity":
                    plt.ylim(0, min(1.05, max(plot_df[metric].max() * 1.1, 0.2)))
            
            plt.suptitle(f"Version Comparison for {model}", fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
            plt.tight_layout(pad=2.0)  # Add extra padding
            plt.savefig(plots_dir / f"{model}_version_comparison.png", bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.error(f"Error generating version comparison plots: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_simplified_correlation_matrix(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate simplified correlation matrix focusing on essential metrics only.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots
    """
    try:
        # Skip if DataFrame is too small
        if df.shape[0] < 3:
            return
            
        # Define the essential metrics for correlation analysis
        essential_metrics = [
            "reasoning_count", 
            "avg_framework_confidence", 
            "framework_agreement", 
            "semantic_similarity"
        ]
        
        # Add any framework accuracy columns if they exist
        for col in df.columns:
            if col.startswith("framework_accuracy_") and not df[col].isna().all():
                essential_metrics.append(col)
                
        # Filter to metrics that exist in the dataframe
        valid_metrics = [m for m in essential_metrics if m in df.columns and not df[m].isna().all()]
        
        if len(valid_metrics) < 2:
            logger.warning("Not enough metrics for correlation matrix")
            return
            
        # Create a DataFrame with only the essential metrics
        essential_df = df[valid_metrics].copy()
        
        # For each framework accuracy metric, filter out null values
        accuracy_metrics = [m for m in valid_metrics if m.startswith("framework_accuracy_")]
        for metric in accuracy_metrics:
            # Create a mask for non-null values in this metric
            non_null_mask = ~essential_df[metric].isna()
            # If all values are null, drop the column
            if non_null_mask.sum() == 0:
                essential_df = essential_df.drop(columns=[metric])
                valid_metrics.remove(metric)
                continue
                
            # For correlation analysis, we need complete rows
            # This means filtering to rows where THIS specific metric is not null
            metric_df = essential_df[non_null_mask].copy()
            
            # Replace the full dataframe with the filtered one for this specific metric
            if metric == accuracy_metrics[0]:
                # For first metric, initialize filtered dataframe
                filtered_df = metric_df
            else:
                # For subsequent metrics, merge with previous filtered dataframe
                # Keep only rows where both metric values are non-null
                common_indices = filtered_df.index.intersection(metric_df.index)
                filtered_df = filtered_df.loc[common_indices]
                
        # If we have a filtered dataframe, use it
        if 'filtered_df' in locals() and not filtered_df.empty:
            essential_df = filtered_df
        
        # Check if we still have enough data after filtering
        if essential_df.shape[0] < 3 or len(valid_metrics) < 2:
            logger.warning("Not enough data for correlation matrix after filtering nulls")
            return
        
        # Calculate correlation matrix
        try:
            # First drop any columns that became all-null after filtering
            for col in essential_df.columns:
                if essential_df[col].isna().all():
                    essential_df = essential_df.drop(columns=[col])
            
            # Drop any rows with nulls to ensure clean correlation
            essential_df = essential_df.dropna()
            
            # Check if we still have enough data
            if essential_df.shape[0] < 3 or essential_df.shape[1] < 2:
                logger.warning("Not enough non-null data for correlation matrix")
                return
                
            corr_matrix = essential_df.corr(method='pearson')
            
            # Replace NaN values with 0 for visualization
            corr_matrix = corr_matrix.fillna(0)
            
            # Create prettier labels for the matrix
            pretty_labels = []
            for metric in corr_matrix.columns:
                if metric.startswith("framework_accuracy_"):
                    label = "Acc: " + metric.replace("framework_accuracy_", "")
                elif metric == "framework_agreement":
                    label = "Framework Agmt"
                elif metric == "semantic_similarity":
                    label = "Semantic Sim"
                elif metric == "reasoning_count":
                    label = "Reasoning Count"
                elif metric == "avg_framework_confidence":
                    label = "Avg Framework Conf"
                else:
                    label = metric.replace("_", " ").title()
                
                # Limit label length
                if len(label) > 20:
                    label = label[:18] + ".."
                    
                pretty_labels.append(label)
                
            # Rename for display
            corr_matrix.columns = pretty_labels
            corr_matrix.index = pretty_labels
            
            # Generate publication-quality heatmap - KEEP IT SIMPLE, with full matrix
            plt.figure(figsize=(10, 8))
            
            # Use a colorblind-friendly diverging palette
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Plot heatmap without mask to show full matrix
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap=cmap,
                vmin=-1.0,
                vmax=1.0, 
                center=0,
                fmt='.2f', 
                linewidths=0.5,
                square=True,  # Make cells square
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
            )
            
            plt.title('Correlation Between Key Metrics', fontweight="bold", pad=20)
            plt.tight_layout()
            
            # Save with high DPI and tight bounding box
            plt.savefig(plots_dir / "essential_metrics_correlation.png", bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Error generating correlation matrix: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"Error generating simplified correlation matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_detailed_prediction_analysis(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate detailed model-by-case prediction success matrices for scientific paper.
    
    Args:
        df: DataFrame with evaluation results
        plots_dir: Directory to save plots and tables
    """
    try:
        # Create a table directory for text-based outputs
        tables_dir = plots_dir / "tables"
        tables_dir.mkdir(exist_ok=True, parents=True)
        
        # Get framework match columns (columns starting with "match_")
        match_columns = [col for col in df.columns if col.startswith("match_")]
        
        if not match_columns:
            logger.warning("No match data found for detailed prediction analysis")
            return
        
        # 1. Generate model success rates by framework
        framework_success = pd.DataFrame()
        
        for col in match_columns:
            framework = col.replace("match_", "")
            # Skip if framework isn't in the data
            if f"gold_{framework}" not in df.columns:
                continue
                
            # Filter out cases where the gold standard is null, None, or "null"
            # First make a copy to avoid modifying the original dataframe
            framework_df = df.copy()
            # Filter null values (string "null", None, np.nan and empty strings)
            framework_df = framework_df[~framework_df[f"gold_{framework}"].isin(["null", "None", "", "Null"])]
            framework_df = framework_df[~framework_df[f"gold_{framework}"].isna()]
            
            # Skip if no valid gold standards exist for this framework
            if len(framework_df) == 0:
                logger.info(f"No valid gold standards for {framework}, skipping...")
                continue
                
            # Group by model and calculate success rate for this framework
            # Convert boolean values to float for consistent handling
            if framework_df[col].dtype == bool:
                framework_df[col] = framework_df[col].astype(float)
            
            # Ensure the column is numeric
            try:
                framework_df[col] = pd.to_numeric(framework_df[col], errors='coerce')
            except:
                logger.warning(f"Column {col} contains non-numeric values. Attempting to convert.")
                
            # Drop any rows with NaN values after conversion
            framework_df = framework_df.dropna(subset=[col])
            
            # Skip if no data left after cleaning
            if len(framework_df) == 0:
                logger.info(f"No valid data for {framework} after filtering non-numeric values")
                continue
                
            group = framework_df.groupby("model")[col].mean().reset_index()
            group = group.rename(columns={col: f"{framework}_success_rate"})
            
            if framework_success.empty:
                framework_success = group
            else:
                framework_success = framework_success.merge(group, on="model", how="outer")
        
        # Sort by overall success rate
        if not framework_success.empty:
            # Add average success across frameworks
            numeric_cols = [col for col in framework_success.columns if col != "model"]
            if numeric_cols:
                framework_success["average_success"] = framework_success[numeric_cols].mean(axis=1)
                framework_success = framework_success.sort_values("average_success", ascending=False)
            
            # Save framework success rates to CSV
            framework_success.to_csv(tables_dir / "framework_success_by_model.csv", index=False)
            
            # Create a visual heatmap of success rates
            plt.figure(figsize=(max(8, len(framework_success) * 0.5), max(6, len(numeric_cols) * 0.5)))
            
            # Prepare data for visualization
            framework_success_plot = framework_success.copy()
            
            # Ensure all data is numeric for the heatmap
            for col in framework_success_plot.columns:
                if col != "model":
                    # Replace NaNs with 0 for visualization
                    framework_success_plot[col] = framework_success_plot[col].fillna(0)
                    # Convert to float to ensure numeric type
                    framework_success_plot[col] = framework_success_plot[col].astype(float)
            
            # Drop average success column for the heatmap
            if "average_success" in framework_success_plot.columns:
                plot_data = framework_success_plot.drop(columns=["average_success"])
            else:
                plot_data = framework_success_plot
                
            # Set model as index for better heatmap
            plot_data = plot_data.set_index("model")
            
            # Final check to ensure numeric data only
            if plot_data.shape[0] > 0 and plot_data.shape[1] > 0:
                # Verify we have numeric data
                if not np.issubdtype(plot_data.values.dtype, np.number):
                    logger.warning("Data contains non-numeric values, attempting final conversion")
                    # Convert any remaining object dtypes to float
                    for col in plot_data.columns:
                        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce').fillna(0).astype(float)
                
                # Generate heatmap
                ax = sns.heatmap(
                    plot_data, 
                    annot=True, 
                    cmap="YlGnBu", 
                    vmin=0, 
                    vmax=1, 
                    fmt=".2f",
                    linewidths=0.5
                )
                plt.title("Model Success Rate by Framework", fontweight="bold", pad=20)
                plt.tight_layout()
                plt.savefig(plots_dir / "model_framework_success_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                logger.warning("No data available for model framework success heatmap")
        
        # 2. Generate case-by-model success matrix - which models got which cases right
        if "test_case" in df.columns:
            # For each framework, create a matrix of test case vs model
            for framework_col in match_columns:
                framework = framework_col.replace("match_", "")
                
                # Skip if no gold standard for this framework
                if f"gold_{framework}" not in df.columns:
                    continue
                
                # Filter out cases where gold standard is null, None, or "null"
                framework_df = df.copy()
                framework_df = framework_df[~framework_df[f"gold_{framework}"].isin(["null", "None", "", "Null"])]
                framework_df = framework_df[~framework_df[f"gold_{framework}"].isna()]
                
                # Skip if no valid gold standards exist for this framework
                if len(framework_df) == 0:
                    logger.info(f"No valid gold standards for {framework}, skipping case-model analysis...")
                    continue
                
                # Ensure match column is numeric
                if framework_df[framework_col].dtype == bool:
                    framework_df[framework_col] = framework_df[framework_col].astype(float)
                
                try:
                    framework_df[framework_col] = pd.to_numeric(framework_df[framework_col], errors='coerce')
                except:
                    logger.warning(f"Column {framework_col} contains non-numeric values. Attempting to convert.")
                
                # Drop rows with NaN after conversion
                framework_df = framework_df.dropna(subset=[framework_col])
                
                # Skip if no data left
                if len(framework_df) == 0:
                    logger.info(f"No valid data for {framework} after filtering non-numeric values")
                    continue
                
                # Create a pivot table showing which models got which cases right
                success_matrix = framework_df.pivot_table(
                    index="test_case", 
                    columns="model", 
                    values=framework_col,
                    aggfunc=lambda x: np.mean(x) if len(x) > 0 else np.nan
                )
                
                # Convert all values to float to ensure numeric type
                success_matrix = success_matrix.astype(float)
                
                # Calculate success rates for cases and models
                success_matrix["case_success_rate"] = success_matrix.mean(axis=1)
                model_success = success_matrix.mean(axis=0)
                
                # Sort by case success rate
                success_matrix = success_matrix.sort_values("case_success_rate", ascending=False)
                
                # Save raw data
                success_matrix.to_csv(tables_dir / f"{framework}_case_model_matrix.csv")
                
                # For each case, also save what the gold standard was
                gold_values = framework_df.groupby("test_case")[f"gold_{framework}"].first()
                gold_values.to_csv(tables_dir / f"{framework}_gold_standards.csv", header=["gold_standard"])
                
                # Check if we have data to plot
                if success_matrix.shape[0] == 0 or success_matrix.shape[1] <= 1:  # Need at least 1 column beyond case_success_rate
                    logger.warning(f"Insufficient data for {framework} case model heatmap")
                    continue
                
                # Create a visual heatmap
                plt.figure(figsize=(max(10, len(success_matrix.columns) * 0.4), 
                                   max(8, len(success_matrix) * 0.4)))
                
                # Drop case success rate column for the heatmap
                plot_data = success_matrix.drop(columns=["case_success_rate"])
                
                # Ensure all data is numeric (float)
                plot_data = plot_data.astype(float)
                
                # Generate heatmap
                ax = sns.heatmap(
                    plot_data, 
                    annot=True, 
                    cmap="YlGnBu", 
                    vmin=0, 
                    vmax=1, 
                    fmt=".2f",
                    linewidths=0.5
                )
                plt.title(f"{framework} Classification Success by Case and Model", fontweight="bold", pad=20)
                plt.tight_layout()
                plt.savefig(plots_dir / f"{framework}_case_model_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Create a detailed color-coded table showing predictions by model and case
                if "classification_" + framework in df.columns:
                    classification_matrix = framework_df.pivot_table(
                        index="test_case", 
                        columns="model", 
                        values="classification_" + framework,
                        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else ""
                    )
                    
                    # Save the detailed classification table
                    classification_matrix.to_csv(tables_dir / f"{framework}_classifications_by_model.csv")
                    
                    # Generate a text report for inclusion in the paper
                    with open(tables_dir / f"{framework}_detailed_report.txt", "w") as f:
                        f.write(f"DETAILED PREDICTION REPORT FOR {framework.upper()}\n")
                        f.write("="*80 + "\n\n")
                        
                        for case in classification_matrix.index:
                            gold_value = gold_values.loc[case] if case in gold_values.index else "UNKNOWN"
                            f.write(f"Case: {case}\n")
                            f.write(f"Gold Standard: {gold_value}\n")
                            f.write("Model Predictions:\n")
                            
                            for model in classification_matrix.columns:
                                prediction = classification_matrix.loc[case, model]
                                success = "✓" if prediction == gold_value else "✗"
                                f.write(f"  {model}: {prediction} {success}\n")
                            
                            f.write("\n" + "-"*80 + "\n\n")
        
        # 4. Generate paper-ready summary tables using HTML for rich formatting
        try:
            # Create a single summary HTML for all frameworks
            with open(tables_dir / "framework_prediction_summary.html", "w") as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Framework Prediction Analysis</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        .success { background-color: #d4edda; }
                        .failure { background-color: #f8d7da; }
                        h2 { margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
                        .summary { font-weight: bold; }
                    </style>
                </head>
                <body>
                    <h1>Framework Prediction Analysis</h1>
                """)
                
                for framework_col in match_columns:
                    framework = framework_col.replace("match_", "")
                    
                    # Skip if no gold standard for this framework
                    if f"gold_{framework}" not in df.columns:
                        continue
                    
                    # Filter out null gold standards for this framework
                    framework_df = df.copy()
                    framework_df = framework_df[~framework_df[f"gold_{framework}"].isin(["null", "None", "", "Null"])]
                    framework_df = framework_df[~framework_df[f"gold_{framework}"].isna()]
                    
                    # Skip if no valid gold standards exist for this framework
                    if len(framework_df) == 0:
                        continue
                    
                    # Ensure match column is numeric/boolean
                    if not pd.api.types.is_bool_dtype(framework_df[framework_col]) and not pd.api.types.is_numeric_dtype(framework_df[framework_col]):
                        try:
                            framework_df[framework_col] = pd.to_numeric(framework_df[framework_col], errors='coerce')
                            framework_df = framework_df.dropna(subset=[framework_col])
                        except:
                            logger.warning(f"Could not convert {framework_col} to numeric for HTML summary")
                    
                    # Skip if no data left
                    if len(framework_df) == 0:
                        continue
                    
                    f.write(f"<h2>{framework} Framework Analysis</h2>")
                    
                    # Model success rates
                    f.write("<h3>Model Success Rates</h3>")
                    f.write("<table>")
                    f.write("<tr><th>Model</th><th>Success Rate</th><th>Correct</th><th>Total</th></tr>")
                    
                    model_results = {}
                    for model in framework_df["model"].unique():
                        model_data = framework_df[framework_df["model"] == model]
                        success_rate = model_data[framework_col].mean()
                        correct = model_data[framework_col].sum()
                        total = len(model_data)
                        model_results[model] = (success_rate, correct, total)
                    
                    # Sort by success rate
                    for model, (rate, correct, total) in sorted(model_results.items(), key=lambda x: x[1][0], reverse=True):
                        f.write(f"<tr><td>{model}</td><td>{rate:.2f}</td><td>{int(correct)}</td><td>{total}</td></tr>")
                    
                    f.write("</table>")
                    
                    # Case success rates
                    if "test_case" in framework_df.columns:
                        f.write("<h3>Case Analysis</h3>")
                        f.write("<table>")
                        f.write("<tr><th>Test Case</th><th>Gold Standard</th><th>Success Rate</th><th>Models with Correct Prediction</th></tr>")
                        
                        case_results = {}
                        for case in framework_df["test_case"].unique():
                            case_data = framework_df[framework_df["test_case"] == case]
                            gold = case_data[f"gold_{framework}"].iloc[0] if not case_data.empty else "UNKNOWN"
                            success_rate = case_data[framework_col].mean()
                            correct_models = case_data[case_data[framework_col] > 0.5]["model"].unique()
                            case_results[case] = (gold, success_rate, correct_models)
                        
                        # Sort by success rate
                        for case, (gold, rate, correct_models) in sorted(case_results.items(), key=lambda x: x[1][1], reverse=True):
                            correct_list = ", ".join(correct_models)
                            f.write(f"<tr><td>{case}</td><td>{gold}</td><td>{rate:.2f}</td><td>{correct_list}</td></tr>")
                        
                        f.write("</table>")
                
                f.write("</body></html>")
                
        except Exception as e:
            logger.error(f"Error generating HTML summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error generating detailed prediction analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())