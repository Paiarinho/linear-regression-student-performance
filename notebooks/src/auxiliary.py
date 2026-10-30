import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Any


import seaborn as sns

def plot_blox_plot(df, columns, target, figsize=(16,4), palette='tab10'):
    fig, axs = plt.subplots(1, len(columns), figsize=figsize, sharey=True)

    for ax, column in zip(axs, columns):
        sns.boxplot(x=column, data=df, ax=ax, showmeans=True, y=target, hue=target, palette=palette)
        ax.set_ylabel("")

    fig.suptitle(target, fontsize=16, fontweight='bold')

    plt.show()


def pairplot(df, pairplot_columns, hue_column=None):
    """
    Generates an optimized pairplot, keeping only the lower triangle.
    """
    g = sns.pairplot(
        data=df,
        vars=pairplot_columns,
        hue=hue_column,
        corner=True,             
        diag_kind="kde",         
        plot_kws={"alpha": 0.6}  
    )
    plt.show()
    return g

# ________________________________________________________________________________________________________________

def check_normality(group_dict: Dict[str, np.ndarray], alpha: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    """
    Checks if all groups follow a normal distribution using the Shapiro-Wilk test.
    Returns an overall boolean and a dictionary with detailed stats per group.
    """
    is_normal_overall = True
    details = {}
    
    for group_name, g in group_dict.items():
        if len(g) >= 3:
            stat, p_val = stats.shapiro(g)
            is_group_normal = bool(p_val >= alpha)
            
            if not is_group_normal:
                is_normal_overall = False
                
            details[group_name] = {
                "stat": stat, 
                "p_val": p_val, 
                "is_normal": is_group_normal
            }
        else:
            is_normal_overall = False
            details[group_name] = {"stat": None, "p_val": None, "is_normal": False}
            
    return is_normal_overall, details

def check_equal_variance(arrays: List[np.ndarray], alpha: float = 0.05) -> Tuple[bool, float, float]:
    """
    Checks if groups have equal variance using Levene's test.
    Returns a boolean, the test statistic, and the p-value.
    """
    stat, p_val = stats.levene(*arrays)
    equal_variance = bool(p_val >= alpha)
    
    return equal_variance, stat, p_val

def evaluate_features_statistics(df: pd.DataFrame, group_cols: List[str], target_col: str, alpha: float = 0.05) -> None:
    """
    Orchestrates the statistical tests for multiple categorical features against a target,
    printing a clean, organized report with all numerical results and test suggestions.
    """
    for col in group_cols:
        # Isolate relevant columns and drop NaNs
        data = df[[col, target_col]].dropna()
        
        # Create a dictionary of {category_name: array_of_scores}
        group_dict = {str(name): group[target_col].values for name, group in data.groupby(col)}
        arrays = list(group_dict.values())
        n_groups = len(arrays)
        
        if n_groups < 2:
            print(f"Skipping '{col}': Needs at least 2 categories.\n")
            continue
            
        # 1. Apply independent checks
        is_normal, norm_details = check_normality(group_dict, alpha)
        equal_var, lev_stat, lev_p = check_equal_variance(arrays, alpha)
        
        # 2. Test Suggestion Routing
        if n_groups == 2:
            if is_normal and equal_var:
                suggestion = "Independent Student's T-test (ttest_ind)"
            elif is_normal and not equal_var:
                suggestion = "Welch's T-test (ttest_ind with equal_var=False)"
            else:
                suggestion = "Mann-Whitney U Test (mannwhitneyu)"
        else:
            if is_normal and equal_var:
                suggestion = "One-way ANOVA (f_oneway)"
            else:
                suggestion = "Kruskal-Wallis H Test (kruskal)"
                
        # 3. Print Formatted Report
        print(f"{'='*50}")
        print(f" FEATURE: {col.upper()} vs {target_col.upper()}")
        print(f"{'='*50}")
        print(f"Number of groups: {n_groups}\n")
        
        print("--- Normality Check (Shapiro-Wilk) ---")
        print(f"Overall Normal: {is_normal}")
        for g_name, res in norm_details.items():
            if res['stat'] is not None:
                print(f"  > Group '{g_name}': stat = {res['stat']:.4f}, p-value = {res['p_val']:.4f} -> Normal: {res['is_normal']}")
            else:
                print(f"  > Group '{g_name}': Not enough data to test.")
        print()
        
        print("--- Variance Check (Levene) ---")
        print(f"Equal Variance: {equal_var}")
        print(f"Statistic = {lev_stat:.4f}, p-value = {lev_p:.4f}\n")
        
        print("--- Suggested Statistical Test ---")
        print(f"-> {suggestion}\n\n")
