"""
FILE: core/statistician_engine.py
-----------------------------------
Pure statistical test execution functions.
No LangChain or LLM dependencies.

One function per test. Each returns a StatisticianOutput.
The fitted model object (for regression tests) is returned separately
so the tools layer can store it in memory for the Model Critic.
"""

import numpy as np
import pandas as pd
from scipy import stats

from Schemas.statistician import (
    StatisticianOutput,
    TestFamily,
    SignificanceVerdict,
    InferenceResult,
    RegressionResult,
    CorrelationResult,
    DimensionalityResult,
    Coefficient,
    PCAComponent,
)

from constants.statistician import DEFAULT_ALPHA


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _verdict(p_value: float, alpha: float = DEFAULT_ALPHA) -> SignificanceVerdict:
    """Determines significance verdict from p-value."""
    if p_value < alpha:
        return SignificanceVerdict.SIGNIFICANT
    elif p_value < alpha * 1.1:
        return SignificanceVerdict.BORDERLINE
    return SignificanceVerdict.NOT_SIGNIFICANT


def _group_stats(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
) -> dict[str, dict]:
    """Computes mean, std, and n per group."""
    result = {}
    for group, data in df.groupby(grouping_var)[dependent_var]:
        clean = data.dropna()
        result[str(group)] = {
            "mean": round(float(clean.mean()), 4),
            "std":  round(float(clean.std()), 4),
            "n":    int(len(clean)),
        }
    return result


def _cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Cohen's d effect size for two-group comparisons."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std() ** 2 + (n2 - 1) * group2.std() ** 2) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return round(float((group1.mean() - group2.mean()) / pooled_std), 4)


def _correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "strong"
    elif abs_r >= 0.3:
        return "moderate"
    return "weak"

def _encode_features(df: pd.DataFrame, independent_vars: list[str]) -> pd.DataFrame:
    """
    Encodes categorical columns as dummy variables for regression and PCA.
    drop_first=True avoids multicollinearity.
    Bool dummies converted to int for statsmodels/sklearn compatibility.
    """
    X = df[independent_vars].copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        bool_cols = X.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            X[bool_cols] = X[bool_cols].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


# ─────────────────────────────────────────────
# INFERENCE TESTS
# ─────────────────────────────────────────────

def run_independent_ttest(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
    correction_type: str | None = None,
) -> tuple[StatisticianOutput, None]:
    """Independent Samples T-Test. Applies Welch's correction if correction_type='welch'."""
    groups = df[grouping_var].dropna().unique()
    g1 = df[df[grouping_var] == groups[0]][dependent_var].dropna()
    g2 = df[df[grouping_var] == groups[1]][dependent_var].dropna()

    equal_var = correction_type != "welch"
    t_stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
    d = _cohens_d(g1, g2)

    verdict = _verdict(p)
    direction = (
        f"Group '{groups[0]}' (mean={round(float(g1.mean()), 2)}) "
        f"{'>' if g1.mean() > g2.mean() else '<'} "
        f"Group '{groups[1]}' (mean={round(float(g2.mean()), 2)})."
    )
    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"difference in '{dependent_var}' between groups "
        f"(t={round(float(t_stat), 4)}, p={round(float(p), 4)}). {direction}"
        + (f" Welch's correction applied (unequal variances)." if correction_type == "welch" else "")
    )

    result = InferenceResult(
        test_name="Independent Samples T-Test",
        statistic=round(float(t_stat), 4),
        statistic_label="t",
        p_value=round(float(p), 4),
        verdict=verdict,
        group_stats=_group_stats(df, dependent_var, grouping_var),
        df=round(float(len(g1) + len(g2) - 2), 4),
        effect_size=d,
        effect_size_label="Cohen's d",
        correction_applied=correction_type,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Independent Samples T-Test",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(df),
        correction_applied=correction_type,
        columns_used={"dependent": dependent_var, "grouping": grouping_var},
    ), None


def run_paired_ttest(
    df: pd.DataFrame,
    dependent_var: str,
    independent_var: str,
) -> tuple[StatisticianOutput, None]:
    """Paired T-Test."""
    col1 = df[dependent_var].dropna()
    col2 = df[independent_var].loc[col1.index].dropna()
    aligned = pd.concat([col1, col2], axis=1).dropna()

    t_stat, p = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1])
    verdict = _verdict(p)

    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"difference between paired measurements "
        f"(t={round(float(t_stat), 4)}, p={round(float(p), 4)})."
    )

    result = InferenceResult(
        test_name="Paired T-Test",
        statistic=round(float(t_stat), 4),
        statistic_label="t",
        p_value=round(float(p), 4),
        verdict=verdict,
        df=float(len(aligned) - 1),
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Paired T-Test",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(aligned),
        columns_used={"variable_1": dependent_var, "variable_2": independent_var},
    ), None


def run_one_sample_ttest(
    df: pd.DataFrame,
    dependent_var: str,
    popmean: float = 0.0,
) -> tuple[StatisticianOutput, None]:
    """One Sample T-Test against a population mean."""
    series = df[dependent_var].dropna()
    t_stat, p = stats.ttest_1samp(series, popmean)
    verdict = _verdict(p)

    interpretation = (
        f"The mean of '{dependent_var}' ({round(float(series.mean()), 4)}) is "
        f"{'significantly' if verdict == SignificanceVerdict.SIGNIFICANT else 'not significantly'} "
        f"different from the population mean ({popmean}) "
        f"(t={round(float(t_stat), 4)}, p={round(float(p), 4)})."
    )

    result = InferenceResult(
        test_name="One Sample T-Test",
        statistic=round(float(t_stat), 4),
        statistic_label="t",
        p_value=round(float(p), 4),
        verdict=verdict,
        df=float(len(series) - 1),
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="One Sample T-Test",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(series),
        columns_used={"variable": dependent_var},
    ), None


def run_one_way_anova(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
) -> tuple[StatisticianOutput, None]:
    """One-Way ANOVA."""
    groups = df[grouping_var].dropna().unique()
    group_data = [df[df[grouping_var] == g][dependent_var].dropna().values for g in groups]

    f_stat, p = stats.f_oneway(*group_data)
    verdict = _verdict(p)

    # Eta-squared effect size
    grand_mean = df[dependent_var].mean()
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in [pd.Series(gd) for gd in group_data]
    )
    ss_total = sum((df[dependent_var] - grand_mean) ** 2)
    eta_sq = round(float(ss_between / ss_total), 4) if ss_total > 0 else None

    df_between = len(groups) - 1
    df_within  = len(df) - len(groups)

    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"difference in '{dependent_var}' across groups "
        f"(F({df_between},{df_within})={round(float(f_stat), 4)}, p={round(float(p), 4)})."
        + (f" Eta-squared={eta_sq} (effect size)." if eta_sq is not None else "")
    )

    result = InferenceResult(
        test_name="One-Way ANOVA",
        statistic=round(float(f_stat), 4),
        statistic_label="F",
        p_value=round(float(p), 4),
        verdict=verdict,
        group_stats=_group_stats(df, dependent_var, grouping_var),
        df_between=float(df_between),
        df_within=float(df_within),
        effect_size=eta_sq,
        effect_size_label="eta-squared",
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="One-Way ANOVA",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(df),
        columns_used={"dependent": dependent_var, "grouping": grouping_var},
    ), None

def run_two_way_anova(df, dependent_var, independent_var1, independent_var2, correction_type=None):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import numpy as np

    formula = (
        f'Q("{dependent_var}") ~ '
        f'C(Q("{independent_var1}")) + '
        f'C(Q("{independent_var2}")) + '
        f'C(Q("{independent_var1}")):C(Q("{independent_var2}"))'
    )

    model       = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract the minimum p-value across all effects (main + interaction)
    p_values  = anova_table["PR(>F)"].dropna()
    min_pval  = float(p_values.min()) if not p_values.empty else 1.0
    f_vals    = anova_table["F"].dropna()
    max_f     = float(f_vals.max()) if not f_vals.empty else 0.0

    # Build a readable interpretation
    rows = []
    for idx, row in anova_table.iterrows():
        f   = row.get("F")
        p   = row.get("PR(>F)")
        if f is None or p is None:
            continue
        sig = "significant" if p < 0.05 else "not significant"
        rows.append(f"{idx}: F={f:.3f}, p={p:.4f} ({sig})")
    interpretation = "Two-Way ANOVA results:\n" + "\n".join(rows)

    # Compute eta-squared for overall model
    ss_model  = anova_table["sum_sq"].dropna().sum()
    ss_resid  = anova_table.loc[anova_table.index.str.lower().str.contains("resid"), "sum_sq"]
    ss_total  = ss_model + (float(ss_resid.iloc[0]) if not ss_resid.empty else 0.0)
    eta_sq    = round(ss_model / ss_total, 4) if ss_total > 0 else None

    verdict = (
        SignificanceVerdict.SIGNIFICANT     if min_pval < 0.05
        else SignificanceVerdict.BORDERLINE if min_pval < 0.10
        else SignificanceVerdict.NOT_SIGNIFICANT
    )

    inference = InferenceResult(
        test_name        = "Two-Way ANOVA",
        statistic        = max_f,
        statistic_label  = "F",
        p_value          = min_pval,
        alpha            = 0.05,
        verdict          = verdict,
        df_between       = float(anova_table["df"].iloc[0]) if len(anova_table) > 0 else None,
        df_within        = float(anova_table["df"].iloc[-1]) if len(anova_table) > 0 else None,
        effect_size      = eta_sq,
        effect_size_label= "eta-squared",
        interpretation   = interpretation,
    )

    output = StatisticianOutput(
        test_name               = "Two-Way ANOVA",
        test_family             = TestFamily.INFERENCE,
        inference_result        = inference,
        n_observations          = len(df),
        columns_used            = {
            "dependent":   dependent_var,
            "independent": f"{independent_var1}, {independent_var2}",
        },
        model_available_in_memory = True,
    )

    return output, model

def run_mann_whitney(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
) -> tuple[StatisticianOutput, None]:
    """Mann-Whitney U Test."""
    groups = df[grouping_var].dropna().unique()
    g1 = df[df[grouping_var] == groups[0]][dependent_var].dropna()
    g2 = df[df[grouping_var] == groups[1]][dependent_var].dropna()

    u_stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    verdict = _verdict(p)

    # r effect size for Mann-Whitney
    n = len(g1) + len(g2)
    z = stats.norm.ppf(p / 2)
    r_effect = round(abs(z) / np.sqrt(n), 4)

    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"difference between groups "
        f"(U={round(float(u_stat), 4)}, p={round(float(p), 4)})."
    )

    result = InferenceResult(
        test_name="Mann-Whitney U Test",
        statistic=round(float(u_stat), 4),
        statistic_label="U",
        p_value=round(float(p), 4),
        verdict=verdict,
        group_stats=_group_stats(df, dependent_var, grouping_var),
        effect_size=r_effect,
        effect_size_label="r",
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Mann-Whitney U Test",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(df),
        columns_used={"dependent": dependent_var, "grouping": grouping_var},
    ), None


def run_wilcoxon(
    df: pd.DataFrame,
    dependent_var: str,
    independent_var: str,
) -> tuple[StatisticianOutput, None]:
    """Wilcoxon Signed-Rank Test."""
    col1 = df[dependent_var].dropna()
    col2 = df[independent_var].loc[col1.index].dropna()
    aligned = pd.concat([col1, col2], axis=1).dropna()

    w_stat, p = stats.wilcoxon(aligned.iloc[:, 0], aligned.iloc[:, 1])
    verdict = _verdict(p)

    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"difference between paired measurements "
        f"(W={round(float(w_stat), 4)}, p={round(float(p), 4)})."
    )

    result = InferenceResult(
        test_name="Wilcoxon Signed-Rank Test",
        statistic=round(float(w_stat), 4),
        statistic_label="W",
        p_value=round(float(p), 4),
        verdict=verdict,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Wilcoxon Signed-Rank Test",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(aligned),
        columns_used={"variable_1": dependent_var, "variable_2": independent_var},
    ), None


def run_kruskal_wallis(
    df: pd.DataFrame,
    dependent_var: str,
    grouping_var: str,
) -> tuple[StatisticianOutput, None]:
    """Kruskal-Wallis Test."""
    groups = df[grouping_var].dropna().unique()
    group_data = [df[df[grouping_var] == g][dependent_var].dropna().values for g in groups]

    h_stat, p = stats.kruskal(*group_data)
    verdict = _verdict(p)

    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"difference across {len(groups)} groups "
        f"(H={round(float(h_stat), 4)}, p={round(float(p), 4)})."
    )

    result = InferenceResult(
        test_name="Kruskal-Wallis Test",
        statistic=round(float(h_stat), 4),
        statistic_label="H",
        p_value=round(float(p), 4),
        verdict=verdict,
        group_stats=_group_stats(df, dependent_var, grouping_var),
        df=float(len(groups) - 1),
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Kruskal-Wallis Test",
        test_family=TestFamily.INFERENCE,
        inference_result=result,
        n_observations=len(df),
        columns_used={"dependent": dependent_var, "grouping": grouping_var},
    ), None


# ─────────────────────────────────────────────
# REGRESSION TESTS
# ─────────────────────────────────────────────

def run_ols_regression(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
    test_name: str = "Simple Linear Regression",
    correction_type: str | None = None,
) -> tuple[StatisticianOutput, object]:
    """
    OLS Regression (SLR or MLR).
    Applies robust standard errors if correction_type='robust_se'.
    Returns (StatisticianOutput, fitted_model) — model stored for Model Critic.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    X = _encode_features(df, independent_vars).dropna()
    y = df[dependent_var].loc[X.index]
    # ── Guard: too many parameters for available rows ──
    n_obs    = len(X)
    n_params = X.shape[1] + 1  # +1 for constant
    if n_obs <= n_params:
        error_msg = (
            f"❌ Not enough observations to fit regression.\n"
            f"Observations: {n_obs}, Parameters: {n_params}\n"
            f"Suggestion: Reduce predictors or collect more data."
        )

        return StatisticianOutput(
            test_name=test_name,
            test_family=TestFamily.REGRESSION,
            regression_result=None,
            n_observations=n_obs,
            correction_applied=correction_type,
            columns_used={
                "dependent": dependent_var,
                "independent": ", ".join(independent_vars)
            },
            model_available_in_memory=False,
            interpretation=error_msg,
        ), None
    X_const = add_constant(X)
    
    model = OLS(y, X_const).fit()

    # Apply correction if requested
    if correction_type == "robust_se":
        model = model.get_robustcov_results(cov_type="HC3")
    elif correction_type == "newey_west":
        model = model.get_robustcov_results(cov_type="HAC", use_t=True, maxlags=1)

    coefficients = []
    ci = np.asarray(model.conf_int())
    
    params = np.asarray(model.params)
    bse = np.asarray(model.bse)
    tvalues = np.asarray(model.tvalues)
    pvalues = np.asarray(model.pvalues)
    for i, name in enumerate(["const"] + independent_vars):
        lower, upper = ci[i]
        coefficients.append(Coefficient(
            variable=name,
            estimate=round(float(params[i]), 6),
            std_error=round(float(bse[i]), 6),
            t_statistic=round(float(tvalues[i]), 4),
            p_value=round(float(pvalues[i]), 4),
            ci_lower=round(float(lower), 6),
            ci_upper=round(float(upper), 6),
        ))

    n_sig = sum(1 for c in coefficients[1:] if c.p_value and c.p_value < DEFAULT_ALPHA)
    interpretation = (
        f"Model R²={round(float(model.rsquared), 4)}, "
        f"Adj. R²={round(float(model.rsquared_adj), 4)}, "
        f"F({int(model.df_model)},{int(model.df_resid)})="
        f"{round(float(model.fvalue), 4)}, p={round(float(model.f_pvalue), 4)}. "
        f"{n_sig} of {len(independent_vars)} predictor(s) are statistically significant."
        + (f" {correction_type.replace('_', ' ').title()} correction applied." if correction_type else "")
    )

    result = RegressionResult(
        test_name=test_name,
        coefficients=coefficients,
        r_squared=round(float(model.rsquared), 4),
        adj_r_squared=round(float(model.rsquared_adj), 4),
        f_statistic=round(float(model.fvalue), 4),
        f_p_value=round(float(model.f_pvalue), 4),
        aic=round(float(model.aic), 4),
        bic=round(float(model.bic), 4),
        rmse=round(float(np.sqrt(model.mse_resid)), 4),
        correction_applied=correction_type,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name=test_name,
        test_family=TestFamily.REGRESSION,
        regression_result=result,
        n_observations=int(model.nobs),
        correction_applied=correction_type,
        columns_used={"dependent": dependent_var,
                      "independent": ", ".join(independent_vars)},
        model_available_in_memory=True,
    ), model


def run_ridge_regression(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
    alpha_reg: float = 1.0,
) -> tuple[StatisticianOutput, object]:
    """Ridge Regression using sklearn. Returns (output, fitted_model)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = _encode_features(df, independent_vars).dropna()
    y = df[dependent_var].loc[X.index]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha_reg)),
    ])
    pipeline.fit(X, y)
    encoded_vars = X.columns.tolist()
    pipeline._aristostat_feature_names = encoded_vars
    ridge_model = pipeline.named_steps["ridge"]
    scaler      = pipeline.named_steps["scaler"]

    coefficients = [
        Coefficient(variable=var, estimate=round(float(coef), 6))
        for var, coef in zip(independent_vars, ridge_model.coef_)
    ]

    y_pred = pipeline.predict(X)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
    rmse = round(float(np.sqrt(ss_res / len(y))), 4)

    interpretation = (
        f"Ridge Regression (alpha={alpha_reg}): R²={r2}, RMSE={rmse}. "
        f"Coefficients are regularised — direct significance testing not available."
    )

    result = RegressionResult(
        test_name="Ridge Regression",
        coefficients=coefficients,
        r_squared=r2,
        rmse=rmse,
        alpha_regularization=alpha_reg,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Ridge Regression",
        test_family=TestFamily.REGRESSION,
        regression_result=result,
        n_observations=len(y),
        columns_used={"dependent": dependent_var,
                      "independent": ", ".join(independent_vars)},
        model_available_in_memory=True,
    ), pipeline


def run_lasso_regression(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
    alpha_reg: float = 1.0,
) -> tuple[StatisticianOutput, object]:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = _encode_features(df, independent_vars).dropna()
    y = df[dependent_var].loc[X.index]

    # ── Use encoded column names, not original — they differ after get_dummies ──
    encoded_vars = X.columns.tolist()

    
    pipeline = Pipeline([
    ("scaler", StandardScaler()),  # scaler normalises coefficients first
    ("lasso", LassoCV(cv=5, max_iter=10000)),
])
    pipeline.fit(X, y)

    lasso_model = pipeline.named_steps["lasso"]
    alpha_reg   = round(float(lasso_model.alpha_), 6)
    pipeline.fit(X, y)
    pipeline._aristostat_feature_names = encoded_vars
    lasso_model = pipeline.named_steps["lasso"]

    # Zip against encoded_vars — correct length match
    coefficients = [
        Coefficient(variable=var, estimate=round(float(coef), 6))
        for var, coef in zip(encoded_vars, lasso_model.coef_)
    ]
    selected = [v for v, c in zip(encoded_vars, lasso_model.coef_) if abs(c) > 1e-6]

    y_pred = pipeline.predict(X)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2   = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
    rmse = round(float(np.sqrt(ss_res / len(y))), 4)

    interpretation = (
        f"Lasso Regression (alpha={alpha_reg}): R²={r2}, RMSE={rmse}. "
        f"{len(selected)} of {len(encoded_vars)} features retained: {selected}."
    )

    result = RegressionResult(
        test_name="Lasso Regression",
        coefficients=coefficients,
        r_squared=r2,
        rmse=rmse,
        alpha_regularization=alpha_reg,
        selected_features=selected,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Lasso Regression",
        test_family=TestFamily.REGRESSION,
        regression_result=result,
        n_observations=len(y),
        columns_used={"dependent": dependent_var,
                      "independent": ", ".join(encoded_vars)},  # encoded names here too
        model_available_in_memory=True,
    ), pipeline

# ─────────────────────────────────────────────
# CORRELATION TESTS
# ─────────────────────────────────────────────

def run_pearson_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> tuple[StatisticianOutput, None]:
    """Pearson Correlation with Fisher Z confidence interval."""
    clean = df[[var1, var2]].dropna()
    r, p = stats.pearsonr(clean[var1], clean[var2])
    verdict = _verdict(p)

    # Fisher Z 95% CI
    n = len(clean)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    ci_lower = round(float(np.tanh(z - 1.96 * se)), 4)
    ci_upper = round(float(np.tanh(z + 1.96 * se)), 4)

    strength = _correlation_strength(r)
    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"{strength} {'positive' if r > 0 else 'negative'} linear relationship "
        f"between '{var1}' and '{var2}' "
        f"(r={round(float(r), 4)}, p={round(float(p), 4)}, 95% CI [{ci_lower}, {ci_upper}])."
    )

    result = CorrelationResult(
        test_name="Pearson Correlation",
        statistic=round(float(r), 4),
        statistic_label="r",
        p_value=round(float(p), 4),
        verdict=verdict,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        correlation_strength=strength,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Pearson Correlation",
        test_family=TestFamily.CORRELATION,
        correlation_result=result,
        n_observations=len(clean),
        columns_used={"variable_1": var1, "variable_2": var2},
    ), None


def run_spearman_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> tuple[StatisticianOutput, None]:
    """Spearman Rank Correlation."""
    clean = df[[var1, var2]].dropna()
    rho, p = stats.spearmanr(clean[var1], clean[var2])
    verdict = _verdict(p)
    strength = _correlation_strength(rho)

    interpretation = (
        f"{'Significant' if verdict == SignificanceVerdict.SIGNIFICANT else 'No significant'} "
        f"{strength} {'positive' if rho > 0 else 'negative'} monotonic relationship "
        f"between '{var1}' and '{var2}' "
        f"(rho={round(float(rho), 4)}, p={round(float(p), 4)})."
    )

    result = CorrelationResult(
        test_name="Spearman Correlation",
        statistic=round(float(rho), 4),
        statistic_label="rho",
        p_value=round(float(p), 4),
        verdict=verdict,
        correlation_strength=strength,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Spearman Correlation",
        test_family=TestFamily.CORRELATION,
        correlation_result=result,
        n_observations=len(clean),
        columns_used={"variable_1": var1, "variable_2": var2},
    ), None


# ─────────────────────────────────────────────
# DIMENSIONALITY
# ─────────────────────────────────────────────

def run_pca(
    df: pd.DataFrame,
    independent_vars: list[str],
) -> tuple[StatisticianOutput, object]:
    """Principal Component Analysis. Standardises data before fitting."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_encoded = _encode_features(df, independent_vars).dropna()
    X_scaled = StandardScaler().fit_transform(X_encoded)
    
    pca = PCA()
    pca.fit(X_scaled)

    cumulative = 0.0
    components = []
    n_for_80 = len(independent_vars)

    for i, (var, pct) in enumerate(zip(
        pca.explained_variance_,
        pca.explained_variance_ratio_
    )):
        cumulative += float(pct)
        loadings = {
            col: round(float(pca.components_[i][j]), 4)
            for j, col in enumerate(independent_vars)
        }
        components.append(PCAComponent(
            component_number=i + 1,
            explained_variance=round(float(var), 4),
            explained_variance_pct=round(float(pct) * 100, 2),
            cumulative_variance_pct=round(cumulative * 100, 2),
            loadings=loadings,
        ))
        if cumulative >= 0.80 and n_for_80 == len(independent_vars):
            n_for_80 = i + 1

    selected = [c for c in components if c.component_number <= n_for_80]
    total_var = round(sum(c.explained_variance_pct for c in selected), 2)

    interpretation = (
        f"PCA on {len(independent_vars)} variables. "
        f"{n_for_80} component(s) explain {total_var}% of total variance (≥80% threshold). "
        f"Top component explains {components[0].explained_variance_pct}% of variance."
    )

    result = DimensionalityResult(
        n_components_total=len(independent_vars),
        n_components_selected=n_for_80,
        components=components,
        variance_explained_80pct=n_for_80,
        total_variance_explained=total_var,
        interpretation=interpretation,
    )
    return StatisticianOutput(
        test_name="Principal Component Analysis",
        test_family=TestFamily.DIMENSIONALITY,
        dimensionality_result=result,
        n_observations=len(X_encoded),
        columns_used={"variables": ", ".join(independent_vars)},
        model_available_in_memory=True,
    ), pca


# ─────────────────────────────────────────────
# DISPATCHER — ROUTES TO CORRECT TEST FUNCTION
# ─────────────────────────────────────────────

def run_test(
    test_name: str,
    df: pd.DataFrame,
    dependent_var: str | None,
    independent_vars: list[str],
    grouping_var: str | None,
    correction_type: str | None = None,
) -> tuple[StatisticianOutput, object | None]:
    """
    Routes to the correct test function based on test_name.
    Returns (StatisticianOutput, fitted_model_or_None).
    """
    if test_name == "Independent Samples T-Test":
        return run_independent_ttest(df, dependent_var, grouping_var, correction_type)

    elif test_name == "Paired T-Test":
        return run_paired_ttest(df, dependent_var, independent_vars[0])

    elif test_name == "One Sample T-Test":
        return run_one_sample_ttest(df, dependent_var)

    elif test_name == "One-Way ANOVA":
        return run_one_way_anova(
            df,
            dependent_var,
            independent_vars[0]
        )

    elif test_name == "Two-Way ANOVA":
        return run_two_way_anova(
            df,
            dependent_var,
            independent_vars[0],
            independent_vars[1]
    )

    elif test_name == "Mann-Whitney U Test":
        return run_mann_whitney(df, dependent_var, grouping_var)

    elif test_name == "Wilcoxon Signed-Rank Test":
        return run_wilcoxon(df, dependent_var, independent_vars[0])

    elif test_name == "Kruskal-Wallis Test":
        return run_kruskal_wallis(df, dependent_var, grouping_var)

    elif test_name == "Simple Linear Regression":
        return run_ols_regression(df, dependent_var, independent_vars, test_name, correction_type)

    elif test_name == "Multiple Linear Regression":
        return run_ols_regression(df, dependent_var, independent_vars, test_name, correction_type)

    elif test_name == "Ridge Regression":
        return run_ridge_regression(df, dependent_var, independent_vars)

    elif test_name == "Lasso Regression":
        return run_lasso_regression(df, dependent_var, independent_vars)

    elif test_name == "Pearson Correlation":
        var2 = independent_vars[0] if independent_vars else grouping_var
        return run_pearson_correlation(df, dependent_var, var2)

    elif test_name == "Spearman Correlation":
        var2 = independent_vars[0] if independent_vars else grouping_var
        return run_spearman_correlation(df, dependent_var, var2)

    elif test_name == "Principal Component Analysis":
        return run_pca(df, independent_vars)

    else:
        raise ValueError(f"Unknown test: '{test_name}'")