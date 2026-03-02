"""
FILE: configs/rectifications.py
---------------------------------
Central registry mapping assumption violation types to proposed solutions.
Used by the Rectification Strategist agent for both pre-test (Assumption Checker)
and post-test (Model Critic) failures — distinguished by a phase flag.

Each solution entry has:
  solution_id:      Unique identifier for this solution
  description:      Plain English description shown to user
  action_type:      What kind of action this solution takes:
                      "transform"    — applies a data transformation to a column
                      "test_switch"  — switches to a different statistical test
                      "correction"   — applies a statistical correction to the test
                      "drop"         — drops problematic rows/columns
  action_details:   Dict with specific parameters for the action:
                      For transform:   {"fn": function_name, "columns": "dependent"|"all"}
                      For test_switch: {"new_test": canonical_test_name}
                      For correction:  {"correction_type": name}
                      For drop:        {"target": "outliers"|"influential_points"}
  next_step:        Where to route after user accepts this solution:
                      "assumption_checker" — re-run assumption checks with new test/data
                      "statistician"       — proceed directly to run the test
  phase:            "pre_test" | "post_test" | "both"
"""

RECTIFICATION_REGISTRY: dict[str, list[dict]] = {

    # ─────────────────────────────────────────────
    # NORMALITY VIOLATIONS
    # ─────────────────────────────────────────────
    "normality": [
        {
            "solution_id":  "normality_log_transform",
            "description":  "Apply a log transformation to the dependent variable to reduce skewness and improve normality.",
            "action_type":  "transform",
            "action_details": {"fn": "log_transform", "columns": "dependent"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "normality_sqrt_transform",
            "description":  "Apply a square root transformation to the dependent variable (milder than log, suitable for moderate skew).",
            "action_type":  "transform",
            "action_details": {"fn": "sqrt_transform", "columns": "dependent"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "normality_switch_mannwhitney",
            "description":  "Switch to Mann-Whitney U Test — a non-parametric alternative that does not assume normality (for 2-group comparisons).",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Mann-Whitney U Test"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "normality_switch_kruskal",
            "description":  "Switch to Kruskal-Wallis Test — a non-parametric alternative that does not assume normality (for 3+ group comparisons).",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Kruskal-Wallis Test"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "normality_switch_spearman",
            "description":  "Switch to Spearman Correlation — a non-parametric alternative that does not assume normality (for correlation analysis).",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Spearman Correlation"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
    ],

    # ─────────────────────────────────────────────
    # NORMALITY OF RESIDUALS (post-test, regression)
    # ─────────────────────────────────────────────
    "normality_of_residuals": [
        {
            "solution_id":  "residuals_log_transform_y",
            "description":  "Apply a log transformation to the dependent variable — often improves residual normality in regression.",
            "action_type":  "transform",
            "action_details": {"fn": "log_transform", "columns": "dependent"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
        {
            "solution_id":  "residuals_sqrt_transform_y",
            "description":  "Apply a square root transformation to the dependent variable.",
            "action_type":  "transform",
            "action_details": {"fn": "sqrt_transform", "columns": "dependent"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
        {
            "solution_id":  "residuals_switch_ridge",
            "description":  "Switch to Ridge Regression — more robust to violations of residual normality with high-dimensional data.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Ridge Regression"},
            "next_step":    "assumption_checker",
            "phase":        "post_test",
        },
    ],

    # ─────────────────────────────────────────────
    # HOMOGENEITY OF VARIANCE (Levene's — pre-test)
    # ─────────────────────────────────────────────
    "homogeneity_of_variance": [
        {
            "solution_id":  "homogeneity_welch_correction",
            "description":  "Apply Welch's correction to the T-Test — this version does not assume equal variances between groups.",
            "action_type":  "correction",
            "action_details": {"correction_type": "welch"},
            "next_step":    "statistician",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "homogeneity_switch_mannwhitney",
            "description":  "Switch to Mann-Whitney U Test — a non-parametric alternative that does not require equal variances.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Mann-Whitney U Test"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "homogeneity_switch_kruskal",
            "description":  "Switch to Kruskal-Wallis Test — non-parametric, does not require equal variances across groups.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Kruskal-Wallis Test"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
    ],

    # ─────────────────────────────────────────────
    # HOMOSCEDASTICITY (Breusch-Pagan — post-test, regression)
    # ─────────────────────────────────────────────
    "homoscedasticity": [
        {
            "solution_id":  "heteroscedasticity_log_transform_y",
            "description":  "Apply a log transformation to the dependent variable — often stabilises variance in regression residuals.",
            "action_type":  "transform",
            "action_details": {"fn": "log_transform", "columns": "dependent"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
        {
            "solution_id":  "heteroscedasticity_robust_se",
            "description":  "Use heteroscedasticity-consistent (robust) standard errors (HC3) — keeps the model but corrects standard errors.",
            "action_type":  "correction",
            "action_details": {"correction_type": "robust_se"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
        {
            "solution_id":  "heteroscedasticity_switch_ridge",
            "description":  "Switch to Ridge Regression — more robust to heteroscedasticity.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Ridge Regression"},
            "next_step":    "assumption_checker",
            "phase":        "post_test",
        },
    ],

    # ─────────────────────────────────────────────
    # MULTICOLLINEARITY (VIF)
    # ─────────────────────────────────────────────
    "no_multicollinearity": [
        {
            "solution_id":  "multicollinearity_switch_ridge",
            "description":  "Switch to Ridge Regression — specifically designed to handle multicollinearity by adding a regularisation penalty.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Ridge Regression"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "multicollinearity_switch_lasso",
            "description":  "Switch to Lasso Regression — performs feature selection by shrinking correlated variable coefficients to zero.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Lasso Regression"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "multicollinearity_switch_pca",
            "description":  "Switch to PCA — reduces correlated variables into uncorrelated principal components before analysis.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Principal Component Analysis"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },{
        "solution_id":  "multicollinearity_drop_variable",
        "description":  "Drop a collinear predictor — remove one of the highly correlated variables from the model and re-run assumption checks. You will be asked which variable to drop.",
        "action_type":  "drop_variable",
        "action_details": {"column": None},   # filled in at runtime from user input
        "next_step":    "assumption_checker",
        "phase":        "pre_test",
    },
    ],

    # ─────────────────────────────────────────────
    # OUTLIERS
    # ─────────────────────────────────────────────
    "no_significant_outliers": [
        {
            "solution_id":  "outliers_drop",
            "description":  "Remove outlier rows identified by the IQR method from the dataset before running the test.",
            "action_type":  "drop",
            "action_details": {"target": "outliers"},
            "next_step":    "assumption_checker",
            "phase":        "both",
        },
        {
            "solution_id":  "outliers_log_transform",
            "description":  "Apply a log transformation to compress extreme values and reduce the influence of outliers.",
            "action_type":  "transform",
            "action_details": {"fn": "log_transform", "columns": "dependent"},
            "next_step":    "assumption_checker",
            "phase":        "both",
        },
        {
            "solution_id":  "outliers_switch_spearman",
            "description":  "Switch to Spearman Correlation — rank-based and less sensitive to outliers than Pearson.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Spearman Correlation"},
            "next_step":    "assumption_checker",
            "phase":        "both",
        },
    ],

    # ─────────────────────────────────────────────
    # AUTOCORRELATION (Durbin-Watson — post-test)
    # ─────────────────────────────────────────────
    "autocorrelation": [
        {
            "solution_id":  "autocorrelation_newey_west",
            "description":  "Apply Newey-West standard errors — corrects for autocorrelation in regression standard errors.",
            "action_type":  "correction",
            "action_details": {"correction_type": "newey_west"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
        {
            "solution_id":  "autocorrelation_differencing",
            "description":  "Apply first-differencing to the dependent variable — removes autocorrelation in time-series data.",
            "action_type":  "transform",
            "action_details": {"fn": "first_difference", "columns": "dependent"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
    ],

    # ─────────────────────────────────────────────
    # INFLUENTIAL POINTS (Cook's Distance — post-test)
    # ─────────────────────────────────────────────
    "influential_points": [
        {
            "solution_id":  "influential_drop",
            "description":  "Remove influential observations (Cook's distance > 4/n) and re-run the model.",
            "action_type":  "drop",
            "action_details": {"target": "influential_points"},
            "next_step":    "statistician",
            "phase":        "post_test",
        },
        {
            "solution_id":  "influential_robust_regression",
            "description":  "Switch to Ridge Regression — more robust to influential points due to regularisation.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Ridge Regression"},
            "next_step":    "assumption_checker",
            "phase":        "post_test",
        },
    ],

    # ─────────────────────────────────────────────
    # LINEARITY
    # ─────────────────────────────────────────────
    "linearity": [
        {
            "solution_id":  "linearity_log_transform_x",
            "description":  "Apply a log transformation to the independent variable to linearise a curved relationship.",
            "action_type":  "transform",
            "action_details": {"fn": "log_transform", "columns": "independent"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "linearity_sqrt_transform_x",
            "description":  "Apply a square root transformation to the independent variable.",
            "action_type":  "transform",
            "action_details": {"fn": "sqrt_transform", "columns": "independent"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "linearity_switch_spearman",
            "description":  "Switch to Spearman Correlation — captures monotonic (not strictly linear) relationships.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Spearman Correlation"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
    ],

    # ─────────────────────────────────────────────
    # SAMPLE SIZE
    # ─────────────────────────────────────────────
    "sufficient_sample_size": [
        {
            "solution_id":  "sample_size_reduce_predictors",
            "description":  "Reduce the number of independent variables to meet the n > 10×p rule — remove less important predictors.",
            "action_type":  "drop",
            "action_details": {"target": "predictors"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
        {
            "solution_id":  "sample_size_switch_ridge",
            "description":  "Switch to Ridge Regression — more stable than OLS with small samples relative to predictor count.",
            "action_type":  "test_switch",
            "action_details": {"new_test": "Ridge Regression"},
            "next_step":    "assumption_checker",
            "phase":        "pre_test",
        },
    ],
}