"""
FILE: core/assumption_engine.py
---------------------------------
Pure statistical functions for checking pre-test assumptions.
No LangChain or LLM dependencies.

Each check function returns an AssumptionResult.
The main run_assumption_checks() function orchestrates all checks
for a given test using the ASSUMPTION_REGISTRY from configs/assumptions.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from Utils.assumptions_requirements_registry import ASSUMPTION_REGISTRY
from Schemas.assumption_checker import (
    AssumptionCheckerOutput,
    AssumptionResult,
    AssumptionStatus,
)
from Schemas.methodologist import MethodologistOutput
from constants.assumption_checker import VIF_THRESHOLD, MIN_GROUP_SIZE, OUTLIER_PCT_THRESHOLD, CORRELATION_THRESHOLD, SKEW_SYMMETRY_THRESHOLD

# ─────────────────────────────────────────────
# INDIVIDUAL CHECK FUNCTIONS
# Each returns an AssumptionResult.
# ─────────────────────────────────────────────

def check_normality_shapiro(
    series: pd.Series,
    assumption: dict,
) -> AssumptionResult:
    """Shapiro-Wilk test for normality on a single series."""
    clean = series.dropna()
    alpha = assumption.get("alpha", 0.05)

    if len(clean) < 3:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Shapiro-Wilk",
            plain_reason="Too few observations (< 3) to test normality.",
        )

    # Shapiro-Wilk is unreliable for n > 5000 — use D'Agostino for large samples
    if len(clean) > 5000:
        stat, p = stats.normaltest(clean)
        test_name = "D'Agostino-Pearson"
    else:
        stat, p = stats.shapiro(clean)
        test_name = "Shapiro-Wilk"

    passed = p >= alpha
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
        test_used=test_name,
        statistic=round(float(stat), 4),
        p_value=round(float(p), 4),
        alpha=alpha,
        plain_reason=(
            f"{test_name}: statistic={round(float(stat), 4)}, p={round(float(p), 4)}. "
            + ("Normality assumption met." if passed
               else f"p < {alpha} — normality assumption violated.")
        ),
    )


def check_normality_shapiro_by_group(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
    assumption: dict,
) -> AssumptionResult:
    """
    Shapiro-Wilk normality test run within each group.
    Fails if ANY group fails normality.
    """
    alpha = assumption.get("alpha", 0.05)
    groups = df[grouping_var].dropna().unique()
    failed_groups = []
    results_detail = []

    for group in groups:
        group_data = df[df[grouping_var] == group][dependent_var].dropna()
        if len(group_data) < 3:
            results_detail.append(f"Group '{group}': too few observations to test.")
            continue

        if len(group_data) > 5000:
            stat, p = stats.normaltest(group_data)
            test_name = "D'Agostino-Pearson"
        else:
            stat, p = stats.shapiro(group_data)
            test_name = "Shapiro-Wilk"

        if p < alpha:
            failed_groups.append(group)
            results_detail.append(
                f"Group '{group}': {test_name} statistic={round(float(stat), 4)}, "
                f"p={round(float(p), 4)} — FAILED."
            )
        else:
            results_detail.append(
                f"Group '{group}': {test_name} statistic={round(float(stat), 4)}, "
                f"p={round(float(p), 4)} — passed."
            )

    passed = len(failed_groups) == 0
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
        test_used="Shapiro-Wilk (per group)",
        alpha=alpha,
        plain_reason=(
            "Normality met in all groups. " if passed
            else f"Normality violated in group(s): {failed_groups}. "
        ) + " | ".join(results_detail),
    )


def check_normality_of_differences(
    series_a: pd.Series,
    series_b: pd.Series,
    assumption: dict,
) -> AssumptionResult:
    """Shapiro-Wilk on the differences between paired observations."""
    alpha = assumption.get("alpha", 0.05)
    differences = (series_a - series_b).dropna()

    if len(differences) < 3:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Shapiro-Wilk (differences)",
            plain_reason="Too few paired observations to test normality of differences.",
        )

    stat, p = stats.shapiro(differences)
    passed = p >= alpha
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
        test_used="Shapiro-Wilk (differences)",
        statistic=round(float(stat), 4),
        p_value=round(float(p), 4),
        alpha=alpha,
        plain_reason=(
            f"Shapiro-Wilk on differences: statistic={round(float(stat), 4)}, "
            f"p={round(float(p), 4)}. "
            + ("Normality of differences met." if passed
               else f"p < {alpha} — normality of differences violated.")
        ),
    )


def check_homoscedasticity_bp(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
    assumption: dict,
) -> AssumptionResult:
    """
    Breusch-Pagan test for homoscedasticity.
    Fits OLS, then tests whether residual variance relates to predictors.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    alpha = assumption.get("alpha", 0.05)

    try:
        X = df[independent_vars].dropna()
        y = df[dependent_var].loc[X.index]
        X_const = add_constant(X)

        model = OLS(y, X_const).fit()
        _, p, _, _ = het_breuschpagan(model.resid, model.model.exog)

        passed = p >= alpha
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
            test_used="Breusch-Pagan",
            p_value=round(float(p), 4),
            alpha=alpha,
            plain_reason=(
                f"Breusch-Pagan: p={round(float(p), 4)}. "
                + ("Homoscedasticity assumption met." if passed
                   else f"p < {alpha} — heteroscedasticity detected.")
            ),
        )
    except Exception as e:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Breusch-Pagan",
            plain_reason=f"Could not run Breusch-Pagan test: {str(e)}",
        )


def check_homogeneity_levene(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
    assumption: dict,
) -> AssumptionResult:
    """Levene's test for equality of variances across groups."""
    alpha = assumption.get("alpha", 0.05)

    groups = df[grouping_var].dropna().unique()
    group_data = [
        df[df[grouping_var] == g][dependent_var].dropna().values
        for g in groups
    ]
    group_data = [g for g in group_data if len(g) >= 2]

    if len(group_data) < 2:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Levene's Test",
            plain_reason="Not enough groups with sufficient data to run Levene's test.",
        )

    stat, p = stats.levene(*group_data)
    passed = p >= alpha
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
        test_used="Levene's Test",
        statistic=round(float(stat), 4),
        p_value=round(float(p), 4),
        alpha=alpha,
        plain_reason=(
            f"Levene's Test: statistic={round(float(stat), 4)}, p={round(float(p), 4)}. "
            + ("Homogeneity of variance met." if passed
               else f"p < {alpha} — unequal variances across groups detected.")
        ),
    )


def check_multicollinearity_vif(
    df: pd.DataFrame,
    independent_vars: list[str],
    assumption: dict,
) -> AssumptionResult:
    """
    Variance Inflation Factor (VIF) check for multicollinearity.
    VIF >= 10 for any variable → multicollinearity problem.
    """
    if len(independent_vars) < 2:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.PASSED,
            test_used="VIF",
            plain_reason="Only one independent variable — multicollinearity not applicable.",
        )

    try:
        X = df[independent_vars].copy()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            bool_cols = X.select_dtypes(include="bool").columns.tolist()
            if bool_cols:
                X[bool_cols] = X[bool_cols].astype(int)
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        # Update independent_vars to encoded column names for VIF loop
        independent_vars = X.columns.tolist()
        vif_scores: dict[str, float] = {}

        for i, col in enumerate(independent_vars):
            others = [c for c in independent_vars if c != col]
            reg = LinearRegression().fit(X[others], X[col])
            r2 = reg.score(X[others], X[col])
            vif = 1 / (1 - r2) if r2 < 1.0 else float("inf")
            vif_scores[col] = round(vif, 4)

        high_vif = {k: v for k, v in vif_scores.items() if v >= VIF_THRESHOLD}
        passed = len(high_vif) == 0

        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
            test_used="VIF",
            plain_reason=(
                f"VIF scores: {vif_scores}. "
                + ("All VIF < 10 — no multicollinearity detected." if passed
                   else f"VIF >= 10 for: {high_vif} — multicollinearity detected.")
            ),
        )
    except Exception as e:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="VIF",
            plain_reason=f"Could not compute VIF: {str(e)}",
        )


def check_outliers_iqr(
    series: pd.Series,
    assumption: dict,
) -> AssumptionResult:
    """IQR-based outlier check. Warning if > 5% of values are outliers."""
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    outliers = clean[(clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)]
    pct = len(outliers) / len(clean) if len(clean) > 0 else 0

    passed = pct <= OUTLIER_PCT_THRESHOLD
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="IQR Method",
        plain_reason=(
            f"{len(outliers)} outlier(s) detected ({pct*100:.1f}% of data). "
            + ("Within acceptable range." if passed
               else f"Exceeds {OUTLIER_PCT_THRESHOLD*100}% threshold — review outliers.")
        ),
    )


def check_sample_size(
    df: pd.DataFrame,
    independent_vars: list[str],
    assumption: dict,
) -> AssumptionResult:
    """Checks n > 10 × number of predictors (rule of thumb)."""
    n = len(df)
    p = len(independent_vars) if independent_vars else 1
    min_required = 10 * p
    passed = n >= min_required

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Sample Size Heuristic",
        plain_reason=(
            f"n={n}, predictors={p}. Minimum recommended: {min_required}. "
            + ("Sample size is adequate." if passed
               else f"Sample size may be insufficient for {p} predictor(s).")
        ),
    )


def check_group_sample_sizes(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
    assumption: dict,
) -> AssumptionResult:
    """Checks that each group has at least MIN_GROUP_SIZE observations."""
    group_counts = df.groupby(grouping_var)[dependent_var].count()
    small_groups = group_counts[group_counts < MIN_GROUP_SIZE]
    passed = len(small_groups) == 0

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Group Size Heuristic",
        plain_reason=(
            f"Group sizes: {group_counts.to_dict()}. "
            + ("All groups have sufficient observations." if passed
               else f"Groups with < {MIN_GROUP_SIZE} observations: {small_groups.to_dict()}.")
        ),
    )


def check_linearity_heuristic(
    series_x: pd.Series,
    series_y: pd.Series,
    assumption: dict,
) -> AssumptionResult:
    """
    Heuristic linearity check using Pearson vs Spearman correlation comparison.
    If Pearson ≈ Spearman → roughly linear.
    If Spearman >> Pearson → non-linear monotonic relationship.
    """
    clean = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
    if len(clean) < 5:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Pearson vs Spearman Heuristic",
            plain_reason="Too few observations to assess linearity.",
        )

    pearson_r, _  = stats.pearsonr(clean["x"], clean["y"])
    spearman_r, _ = stats.spearmanr(clean["x"], clean["y"])
    diff = abs(abs(spearman_r) - abs(pearson_r))

    passed = diff < 0.15  # threshold for "close enough"
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Pearson vs Spearman Heuristic",
        plain_reason=(
            f"Pearson r={round(float(pearson_r), 4)}, "
            f"Spearman r={round(float(spearman_r), 4)}, "
            f"difference={round(diff, 4)}. "
            + ("Relationship appears approximately linear." if passed
               else "Notable difference between Pearson and Spearman — possible non-linearity.")
        ),
    )


def check_dtype_continuous(
    col_name: str,
    profiler_output: dict,
    assumption: dict,
) -> AssumptionResult:
    """Checks that a column is listed as continuous in the profiler output."""
    continuous = {c["column"] for c in profiler_output.get("continuous_columns", [])}
    passed = col_name in continuous

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
        test_used="Dtype Check",
        plain_reason=(
            f"'{col_name}' is continuous — requirement met." if passed
            else f"'{col_name}' is not continuous — this assumption is violated."
        ),
    )


def check_correlation_matrix(
    df: pd.DataFrame,
    independent_vars: list[str],
    assumption: dict,
) -> AssumptionResult:
    """
    For PCA: checks that at least one pair of variables has |correlation| > threshold.
    PCA is not meaningful if all variables are uncorrelated.
    """
    if len(independent_vars) < 2:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Correlation Matrix",
            plain_reason="Need at least 2 variables to check correlation for PCA.",
        )

    corr_matrix = df[independent_vars].corr().abs()
    # Exclude diagonal
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = float(corr_matrix.max().max())
    passed = max_corr >= CORRELATION_THRESHOLD

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Correlation Matrix",
        plain_reason=(
            f"Maximum pairwise correlation: {round(max_corr, 4)}. "
            + ("Sufficient correlation for PCA." if passed
               else f"Low correlations detected (max={round(max_corr, 4)}) — PCA may not be meaningful.")
        ),
    )


def check_distribution_shape_similarity(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
    assumption: dict,
) -> AssumptionResult:
    """
    For Mann-Whitney: checks that both groups have similarly shaped distributions
    by comparing skewness values.
    """
    groups = df[grouping_var].dropna().unique()
    if len(groups) != 2:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Skewness Comparison",
            plain_reason="Expected exactly 2 groups for distribution shape comparison.",
        )

    skews = []
    for g in groups:
        data = df[df[grouping_var] == g][dependent_var].dropna()
        skews.append(float(data.skew()))

    skew_diff = abs(skews[0] - skews[1])
    passed = skew_diff < 1.0

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Skewness Comparison",
        plain_reason=(
            f"Group skewness values: {[round(s, 4) for s in skews]}, "
            f"difference={round(skew_diff, 4)}. "
            + ("Similar distribution shapes." if passed
               else "Notably different distribution shapes — Mann-Whitney result is for stochastic dominance, not median comparison.")
        ),
    )


def check_symmetry_of_differences(
    series_a: pd.Series,
    series_b: pd.Series,
    assumption: dict,
) -> AssumptionResult:
    """For Wilcoxon: checks that the differences between pairs are symmetric."""
    differences = (series_a - series_b).dropna()
    skewness = float(differences.skew())
    passed = abs(skewness) < SKEW_SYMMETRY_THRESHOLD

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Skewness of Differences",
        plain_reason=(
            f"Skewness of differences: {round(skewness, 4)}. "
            + ("Approximately symmetric." if passed
               else f"Absolute skewness > {SKEW_SYMMETRY_THRESHOLD} — distribution of differences may not be symmetric.")
        ),
    )


def check_monotonic_relationship(
    series_x: pd.Series,
    series_y: pd.Series,
    assumption: dict,
) -> AssumptionResult:
    """For Spearman: checks for a monotonic relationship using Spearman correlation."""
    clean = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
    if len(clean) < 5:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Spearman Correlation",
            plain_reason="Too few observations to assess monotonicity.",
        )

    spearman_r, p = stats.spearmanr(clean["x"], clean["y"])
    passed = abs(spearman_r) >= 0.1  # any meaningful monotonic trend

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING,
        test_used="Spearman Correlation",
        statistic=round(float(spearman_r), 4),
        p_value=round(float(p), 4),
        plain_reason=(
            f"Spearman r={round(float(spearman_r), 4)}, p={round(float(p), 4)}. "
            + ("Monotonic relationship detected." if passed
               else "Very weak monotonic relationship — Spearman may not be meaningful.")
        ),
    )


# ─────────────────────────────────────────────
# MANUAL ASSUMPTION RESULT
# ─────────────────────────────────────────────

def make_manual_result(assumption: dict) -> AssumptionResult:
    """
    Creates an AssumptionResult for assumptions that cannot be
    verified programmatically. Marked as MANUAL — user must confirm.
    """
    manual_questions = {
        "independence_of_observations":
            "Can you confirm that your observations are independent of each other? "
            "(e.g. each row is a separate individual, not repeated measurements of the same subject)",
        "paired_observations":
            "Can you confirm that your data consists of matched pairs? "
            "(e.g. before/after measurements on the same subject)",
    }

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.MANUAL,
        manual_confirmation_required=True,
        plain_reason="This assumption cannot be verified programmatically — requires user confirmation.",
        manual_question=manual_questions.get(
            assumption["name"],
            f"Please confirm whether the following assumption holds: {assumption['description']}"
        ),
    )


# ─────────────────────────────────────────────
# MAIN — RUN ALL ASSUMPTION CHECKS
# ─────────────────────────────────────────────

def run_assumption_checks(
    methodologist_output: MethodologistOutput,
    cleaned_df: pd.DataFrame,
    profiler_output: dict,
) -> AssumptionCheckerOutput:
    """
    Runs all pre-test assumption checks for the selected test.
    Uses ASSUMPTION_REGISTRY to look up which checks to run.

    Returns a fully populated AssumptionCheckerOutput.
    """
    test_name     = methodologist_output.selected_test
    dep_var       = methodologist_output.dependent_variable
    ind_vars      = methodologist_output.independent_variables
    grp_var       = methodologist_output.grouping_variable

    assumptions   = ASSUMPTION_REGISTRY.get(test_name, [])
    results: list[AssumptionResult] = []

    for assumption in assumptions:
        fn_name      = assumption.get("test_fn")
        check_method = assumption.get("check_method")

        # ── Manual check — cannot verify programmatically ──
        if check_method == "manual" or fn_name is None:
            results.append(make_manual_result(assumption))
            continue

        # ── Dispatch to the appropriate check function ──
        try:
            result = _dispatch_check(
                fn_name=fn_name,
                assumption=assumption,
                cleaned_df=cleaned_df,
                profiler_output=profiler_output,
                dep_var=dep_var,
                ind_vars=ind_vars,
                grp_var=grp_var,
            )
            results.append(result)
        except Exception as e:
            results.append(AssumptionResult(
                name=assumption["name"],
                description=assumption["description"],
                status=AssumptionStatus.WARNING,
                plain_reason=f"Check could not be completed: {str(e)}",
            ))

    # ── Aggregate counts ──
    passed_count  = sum(1 for r in results if r.status == AssumptionStatus.PASSED)
    failed_count  = sum(1 for r in results if r.status == AssumptionStatus.FAILED)
    warning_count = sum(1 for r in results if r.status == AssumptionStatus.WARNING)
    manual_count  = sum(1 for r in results if r.status == AssumptionStatus.MANUAL)

    pending_manual = [r.name for r in results if r.status == AssumptionStatus.MANUAL]
    has_failures   = failed_count > 0

    # Proceed only if no hard failures
    # (manual checks are pending confirmation, not failures)
    proceed = not has_failures

    return AssumptionCheckerOutput(
        selected_test=test_name,
        results=results,
        total_assumptions=len(results),
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        manual_count=manual_count,
        all_assumptions_met=not has_failures,
        has_failures=has_failures,
        pending_manual_confirmations=pending_manual,
        proceed_to_statistician=proceed,
        summary_message=_build_summary_message(
            test_name, results, passed_count,
            failed_count, warning_count, manual_count
        ),
    )


# ─────────────────────────────────────────────
# PRIVATE — DISPATCH CHECK BY FUNCTION NAME
# ─────────────────────────────────────────────

def _dispatch_check(
    fn_name: str,
    assumption: dict,
    cleaned_df: pd.DataFrame,
    profiler_output: dict,
    dep_var: str | None,
    ind_vars: list[str],
    grp_var: str | None,
) -> AssumptionResult:
    """Routes each check to the correct function with the right arguments."""

    if fn_name == "check_normality_shapiro":
        series = cleaned_df[dep_var] if dep_var else cleaned_df[ind_vars[0]]
        return check_normality_shapiro(series, assumption)

    elif fn_name == "check_normality_shapiro_by_group":
        # Two-Way ANOVA: grp_var is None, factors are in ind_vars
        # Run normality check grouped by the interaction of both factors
        effective_grp = grp_var
        if effective_grp is None and len(ind_vars) >= 1:
            effective_grp = ind_vars[0]  # check normality within first factor's groups
        if effective_grp is None:
            raise ValueError("No grouping variable available for normality by group check.")
        return check_normality_shapiro_by_group(cleaned_df, dep_var, effective_grp, assumption)

    elif fn_name == "check_normality_of_differences":
        return check_normality_of_differences(
            cleaned_df[dep_var], cleaned_df[ind_vars[0]], assumption
        )

    elif fn_name == "check_homoscedasticity_bp":
        ivars = ind_vars if ind_vars else ([grp_var] if grp_var else [])
        return check_homoscedasticity_bp(cleaned_df, dep_var, ivars, assumption)

    elif fn_name == "check_homogeneity_levene":
        # Two-Way ANOVA: create a combined group column for Levene's test
        effective_grp = grp_var
        if effective_grp is None and len(ind_vars) >= 2:
            # Combine both factors into one interaction column
            combined_col = "__twoway_group__"
            cleaned_df = cleaned_df.copy()
            cleaned_df[combined_col] = (
                cleaned_df[ind_vars[0]].astype(str) + "_" +
                cleaned_df[ind_vars[1]].astype(str)
            )
            effective_grp = combined_col
        elif effective_grp is None and len(ind_vars) == 1:
            effective_grp = ind_vars[0]
        if effective_grp is None:
            raise ValueError("No grouping variable available for Levene's test.")
        return check_homogeneity_levene(cleaned_df, dep_var, effective_grp, assumption)

    elif fn_name == "check_multicollinearity_vif":
        return check_multicollinearity_vif(cleaned_df, ind_vars, assumption)

    elif fn_name == "check_outliers_iqr":
        series = cleaned_df[dep_var] if dep_var else cleaned_df[ind_vars[0]]
        return check_outliers_iqr(series, assumption)

    elif fn_name == "check_sample_size":
        return check_sample_size(cleaned_df, ind_vars, assumption)

    elif fn_name == "check_group_sample_sizes":
        effective_grp = grp_var
        if effective_grp is None and len(ind_vars) >= 1:
            effective_grp = ind_vars[0]
        if effective_grp is None:
            raise ValueError("No grouping variable for sample size check.")
        return check_group_sample_sizes(cleaned_df, dep_var, effective_grp, assumption)

    elif fn_name == "check_linearity_heuristic":
        x_col = ind_vars[0] if ind_vars else grp_var
        return check_linearity_heuristic(cleaned_df[x_col], cleaned_df[dep_var], assumption)

    elif fn_name == "check_dtype_continuous":
        col = dep_var or (ind_vars[0] if ind_vars else None)
        return check_dtype_continuous(col, profiler_output, assumption)

    elif fn_name == "check_correlation_matrix":
        return check_correlation_matrix(cleaned_df, ind_vars, assumption)

    elif fn_name == "check_distribution_shape_similarity":
        return check_distribution_shape_similarity(cleaned_df, dep_var, grp_var, assumption)

    elif fn_name == "check_symmetry_of_differences":
        return check_symmetry_of_differences(
            cleaned_df[dep_var], cleaned_df[ind_vars[0]], assumption
        )

    elif fn_name == "check_monotonic_relationship":
        x_col = ind_vars[0] if ind_vars else None
        return check_monotonic_relationship(cleaned_df[x_col], cleaned_df[dep_var], assumption)

    else:
        raise ValueError(f"Unknown check function: {fn_name}")


# ─────────────────────────────────────────────
# PRIVATE — BUILD SUMMARY MESSAGE
# ─────────────────────────────────────────────

def _build_summary_message(
    test_name: str,
    results: list[AssumptionResult],
    passed: int,
    failed: int,
    warnings: int,
    manual: int,
) -> str:
    lines = [f"Assumption check results for **{test_name}**:"]
    lines.append(f"✅ Passed: {passed}  ❌ Failed: {failed}  ⚠️ Warnings: {warnings}  🔵 Manual: {manual}")
    lines.append("")

    for r in results:
        icon = {"passed": "✅", "failed": "❌", "warning": "⚠️", "manual": "🔵"}.get(r.status, "•")
        lines.append(f"{icon} **{r.name}**: {r.plain_reason}")

    return "\n".join(lines)