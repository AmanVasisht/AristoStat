
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

VIF_THRESHOLD            = 10.0   # VIF above this → multicollinearity problem
MIN_GROUP_SIZE           = 5      # minimum observations per group
OUTLIER_PCT_THRESHOLD    = 0.05   # > 5% outliers → warning
CORRELATION_THRESHOLD    = 0.3    # at least one pair must exceed this for PCA to be meaningful
SKEW_SYMMETRY_THRESHOLD  = 1.0    # absolute skewness above this → not symmetric

TEST_VARIABLE_CONSTRAINTS = {
    "Simple Linear Regression":  {"max_independent": 1},
    "One-Sample T-Test":         {"max_independent": 0},
    "Paired T-Test":             {"max_independent": 1},
    "Wilcoxon Signed-Rank":      {"max_independent": 1},
    "Pearson Correlation":       {"max_independent": 1},
    "Spearman Correlation":      {"max_independent": 1},
}