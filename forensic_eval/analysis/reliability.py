"""Reliability analysis functions for profile evaluation."""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_inter_model_agreement(df: pd.DataFrame, categorical_column: str, 
                                  test_case_column: str = 'test_case') -> Dict[str, Any]:
    """Calculate agreement between models on categorical classifications.
    
    Args:
        df: DataFrame containing evaluation results
        categorical_column: Name of the column with categorical data to compare
        test_case_column: Name of the column containing test case identifiers
        
    Returns:
        Dictionary with agreement metrics
    """
    try:
        # Check if required columns exist
        if categorical_column not in df.columns:
            return {"error": f"Column '{categorical_column}' not found in the dataframe"}
            
        if test_case_column not in df.columns:
            return {"error": f"Column '{test_case_column}' not found in the dataframe"}
            
        if 'model' not in df.columns:
            return {"error": "Column 'model' not found in the dataframe"}
        
        # Get unique models and test cases
        models = df['model'].unique()
        test_cases = df[test_case_column].unique()
        
        if len(models) < 2:
            return {"error": "Need at least two models to calculate agreement"}
            
        agreement_results = {}
        overall_kappas = []
        
        # Calculate pairwise agreement for each test case
        for test_case in test_cases:
            test_df = df[df[test_case_column] == test_case]
            
            # Skip test cases with no data
            if len(test_df) < 2:
                continue
                
            # Skip test cases with only one model
            if test_df['model'].nunique() < 2:
                continue
            
            # Check if categorical column has any non-null values
            if test_df[categorical_column].isna().all():
                continue
                
            # Create pivot table
            try:
                test_pivoted = test_df.pivot(index=test_case_column, 
                                           columns='model', 
                                           values=categorical_column)
            except Exception as e:
                logger.warning(f"Error creating pivot table for test case {test_case}: {e}")
                continue
            
            # Calculate pairwise agreement for this test case
            case_results = {}
            case_kappas = []
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    # Skip if either model is missing data for this test case
                    if model1 not in test_pivoted.columns or model2 not in test_pivoted.columns:
                        continue
                        
                    # Get classifications
                    y1 = test_pivoted[model1].dropna()
                    y2 = test_pivoted[model2].dropna()
                    
                    # Only consider rows with values for both models
                    common_idx = y1.index.intersection(y2.index)
                    if len(common_idx) < 2:
                        continue
                        
                    y1 = y1.loc[common_idx]
                    y2 = y2.loc[common_idx]
                    
                    # Skip if all values are the same (Cohen's kappa is undefined)
                    if y1.nunique() < 2 and y2.nunique() < 2:
                        # Check if they agree on the constant value
                        if y1.iloc[0] == y2.iloc[0]:
                            case_results[f"{model1}_vs_{model2}"] = {
                                "kappa": 1.0,  # Perfect agreement
                                "agreement_level": "perfect (constant values)",
                                "sample_size": len(common_idx),
                                "note": "All values are identical"
                            }
                            case_kappas.append(1.0)
                        continue
                    
                    # Calculate Cohen's kappa
                    try:
                        kappa = cohen_kappa_score(y1, y2)
                        
                        # Check for NaN
                        if np.isnan(kappa):
                            continue
                            
                        # Add to results
                        case_results[f"{model1}_vs_{model2}"] = {
                            "kappa": float(kappa),
                            "agreement_level": _interpret_kappa(kappa),
                            "sample_size": len(common_idx)
                        }
                        case_kappas.append(kappa)
                    except Exception as e:
                        logger.warning(f"Error calculating kappa for {model1} vs {model2}: {e}")
                        continue
            
            # Calculate average agreement for this test case
            if case_kappas:
                agreement_results[test_case] = {
                    "pairwise": case_results,
                    "average_kappa": float(np.mean(case_kappas)),
                    "average_agreement_level": _interpret_kappa(np.mean(case_kappas))
                }
                overall_kappas.extend(case_kappas)
        
        # Calculate overall average agreement
        if overall_kappas:
            overall_agreement = {
                "average_kappa": float(np.mean(overall_kappas)),
                "agreement_level": _interpret_kappa(np.mean(overall_kappas)),
                "number_of_comparisons": len(overall_kappas)
            }
        else:
            overall_agreement = {"error": "No valid comparisons found"}
        
        return {
            "metric": categorical_column,
            "test_cases": agreement_results,
            "overall": overall_agreement
        }
    except Exception as e:
        logger.error(f"Error calculating inter-model agreement: {e}")
        return {"error": str(e)}

def _interpret_kappa(kappa: float) -> str:
    """Interpret kappa value according to standard interpretation.
    
    Args:
        kappa: Cohen's kappa value
        
    Returns:
        String interpretation of agreement level
    """
    if kappa < 0:
        return "poor"
    elif kappa < 0.2:
        return "slight"
    elif kappa < 0.4:
        return "fair"
    elif kappa < 0.6:
        return "moderate"
    elif kappa < 0.8:
        return "substantial"
    else:
        return "almost perfect" 