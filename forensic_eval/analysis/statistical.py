"""Statistical analysis functions for profile evaluation."""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

def perform_anova(df: pd.DataFrame, metric: str, group_by: str = 'model') -> Dict[str, Any]:
    """Perform one-way ANOVA on a metric across groups.
    
    Args:
        df: DataFrame containing evaluation results
        metric: Column name of the metric to analyze
        group_by: Column name to group by (default: 'model')
        
    Returns:
        Dictionary with ANOVA results
    """
    try:
        # Create groups based on the group_by column
        groups = []
        group_names = []
        
        for name, group in df.groupby(group_by):
            if metric in group.columns and not group[metric].isna().all():
                values = group[metric].dropna().values
                # Only add if there's variation in the data
                if len(values) > 1 and np.std(values) > 0:
                    groups.append(values)
                    group_names.append(name)
        
        if len(groups) < 2:
            return {"error": "Not enough groups with valid data for ANOVA"}
        
        # Check if all groups have at least some variation
        all_constant = True
        for group in groups:
            if np.std(group) > 0:
                all_constant = False
                break
                
        if all_constant:
            return {"error": "All groups have constant values, ANOVA is not appropriate"}
        
        # Perform ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Check for NaN or infinite values
            if math.isnan(f_stat) or math.isinf(f_stat) or math.isnan(p_value) or math.isinf(p_value):
                return {"error": "ANOVA produced invalid result (NaN or infinite)"}
                
            return {
                "test": "one-way ANOVA",
                "metric": metric,
                "grouped_by": group_by,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "groups": group_names
            }
        except Exception as e:
            return {"error": f"ANOVA computation error: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error performing ANOVA: {e}")
        return {"error": str(e)}

def calculate_effect_size(df: pd.DataFrame, metric: str, group_by: str = 'model') -> Dict[str, Any]:
    """Calculate effect size (Cohen's d) between groups.
    
    Args:
        df: DataFrame containing evaluation results
        metric: Column name of the metric to analyze
        group_by: Column name to group by (default: 'model')
        
    Returns:
        Dictionary with effect size results
    """
    try:
        # Get unique groups
        groups = df[group_by].unique()
        
        if len(groups) < 2:
            return {"error": "Need at least two groups to calculate effect size"}
            
        comparisons = {}
        
        # Calculate Cohen's d for each pair of groups
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                data1 = df[df[group_by] == group1][metric].dropna()
                data2 = df[df[group_by] == group2][metric].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                # Cohen's d calculation
                mean1, mean2 = data1.mean(), data2.mean()
                n1, n2 = len(data1), len(data2)
                
                # Pooled standard deviation
                s1, s2 = data1.std(), data2.std()
                
                # Check if both standard deviations are zero
                if s1 == 0 and s2 == 0:
                    # Both distributions are constant, so there's no effect
                    comparisons[f"{group1}_vs_{group2}"] = {
                        "cohen_d": 0.0,
                        "effect_size": "none (constant values)"
                    }
                    continue
                
                pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                
                # Avoid division by zero
                if pooled_std == 0:
                    # Use the maximum of the means to avoid division by zero
                    max_mean = max(abs(mean1), abs(mean2))
                    if max_mean == 0:
                        d = 0.0  # No difference between means
                    else:
                        d = 1.0 if mean1 != mean2 else 0.0  # Different means but no variation
                else:
                    d = abs(mean1 - mean2) / pooled_std
                
                effect_level = "small" if d < 0.5 else "medium" if d < 0.8 else "large"
                
                comparisons[f"{group1}_vs_{group2}"] = {
                    "cohen_d": float(d),
                    "effect_size": effect_level
                }
        
        if not comparisons:
            return {"error": "No valid comparisons could be made"}
            
        return {
            "metric": metric,
            "comparisons": comparisons
        }
    except Exception as e:
        logger.error(f"Error calculating effect size: {e}")
        return {"error": str(e)}

def calculate_confidence_intervals(df: pd.DataFrame, metric: str, group_by: str = 'model', 
                                 confidence: float = 0.95) -> Dict[str, Any]:
    """Calculate confidence intervals for a metric across groups.
    
    Args:
        df: DataFrame containing evaluation results
        metric: Column name of the metric to analyze
        group_by: Column name to group by (default: 'model')
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Dictionary with confidence interval results
    """
    try:
        intervals = {}
        
        for name, group in df.groupby(group_by):
            data = group[metric].dropna()
            
            if len(data) < 2:
                intervals[name] = {"error": "Not enough data points for confidence interval"}
                continue
            
            # Calculate mean and standard error
            mean = data.mean()
            std_err = stats.sem(data)
            
            # If standard error is zero, we can't calculate a CI
            if std_err == 0:
                intervals[name] = {
                    "mean": float(mean),
                    "lower_ci": float(mean),
                    "upper_ci": float(mean),
                    "ci_width": 0.0,
                    "sample_size": len(data),
                    "note": "All values are identical, confidence interval is a point"
                }
                continue
                
            # Calculate confidence interval
            try:
                ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
                
                # Check for NaN or infinite values
                if any(math.isnan(x) or math.isinf(x) for x in ci):
                    intervals[name] = {
                        "error": "Invalid confidence interval (NaN or infinite values)",
                        "mean": float(mean) if not math.isnan(mean) and not math.isinf(mean) else None,
                        "sample_size": len(data)
                    }
                    continue
                    
                intervals[name] = {
                    "mean": float(mean),
                    "lower_ci": float(ci[0]),
                    "upper_ci": float(ci[1]),
                    "ci_width": float(ci[1] - ci[0]),
                    "sample_size": len(data)
                }
            except Exception as e:
                intervals[name] = {
                    "error": f"Error calculating CI: {str(e)}",
                    "mean": float(mean),
                    "sample_size": len(data)
                }
        
        if not intervals:
            return {"error": "No valid confidence intervals could be calculated"}
            
        return {
            "metric": metric,
            "confidence_level": confidence,
            "intervals": intervals
        }
    except Exception as e:
        logger.error(f"Error calculating confidence intervals: {e}")
        return {"error": str(e)} 