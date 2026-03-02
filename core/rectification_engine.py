"""
FILE: core/rectification_engine.py
------------------------------------
Pure logic for the Rectification Strategist agent.
No LangChain or LLM dependencies.

Responsibilities:
  1. Look up proposed solutions for each failed assumption from RECTIFICATION_REGISTRY
  2. Filter solutions by phase (pre_test vs post_test)
  3. Apply chosen solution — transform data, switch test, or apply correction
  4. Track rectification attempts to prevent infinite loops
"""

import numpy as np
import pandas as pd

from Utils.rectification_requirements_registry import RECTIFICATION_REGISTRY
from Schemas.rectification_strategist import (
    RectificationOutput,
    RectificationPhase,
    NextStep,
    ProposedSolution,
    AppliedTransform,
)


# ─────────────────────────────────────────────
# HELPER — GET PROPOSALS FOR FAILED ASSUMPTIONS
# ─────────────────────────────────────────────

def get_proposals_for_failures(
    failed_assumptions: list[str],
    phase: RectificationPhase,
) -> list[ProposedSolution]:
    """
    Looks up proposed solutions for each failed assumption.
    Filters by phase (pre_test / post_test / both).
    Deduplicates solutions that appear for multiple failures
    (e.g. log transform suggested for both normality and outliers).

    Returns a ranked, deduplicated list of ProposedSolution objects.
    Test switches are listed before transforms (less invasive first).
    """
    seen_ids: set[str] = set()
    test_switches: list[ProposedSolution] = []
    transforms: list[ProposedSolution] = []
    corrections: list[ProposedSolution] = []
    drops: list[ProposedSolution] = []

    for assumption_name in failed_assumptions:
        solutions = RECTIFICATION_REGISTRY.get(assumption_name, [])
        for sol in solutions:
            # Filter by phase
            sol_phase = sol.get("phase", "both")
            if sol_phase != "both" and sol_phase != phase.value:
                continue

            sol_id = sol["solution_id"]
            if sol_id in seen_ids:
                continue
            seen_ids.add(sol_id)

            proposal = ProposedSolution(
                solution_id=sol_id,
                description=sol["description"],
                action_type=sol["action_type"],
                action_details=sol.get("action_details", {}),
                next_step=sol.get("next_step", "assumption_checker"),
            )

            # Sort by action type — corrections first, then test switches, transforms, drops
            if sol["action_type"] == "correction":
                corrections.append(proposal)
            elif sol["action_type"] == "test_switch":
                test_switches.append(proposal)
            elif sol["action_type"] == "transform":
                transforms.append(proposal)
            elif sol["action_type"] == "drop":
                drops.append(proposal)

    # Order: corrections → test_switches → transforms → drops
    return corrections + test_switches + transforms + drops


# ─────────────────────────────────────────────
# TRANSFORM FUNCTIONS
# ─────────────────────────────────────────────

def apply_log_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[AppliedTransform]]:
    """
    Applies log1p transformation (log(x+1)) to handle zero values safely.
    Only applied to positive-valued columns — skips if any value <= -1.
    """
    df = df.copy()
    applied: list[AppliedTransform] = []

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if (series <= -1).any():
            continue  # log1p not safe for values <= -1

        orig_min = round(float(series.min()), 4)
        orig_max = round(float(series.max()), 4)
        df[col] = np.log1p(df[col])
        applied.append(AppliedTransform(
            column=col,
            transform_type="log_transform",
            original_min=orig_min,
            original_max=orig_max,
            note="log1p (log(x+1)) applied to handle zero values safely.",
        ))

    return df, applied


def apply_sqrt_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[AppliedTransform]]:
    """
    Applies square root transformation.
    Only applied to non-negative columns — skips if any value < 0.
    """
    df = df.copy()
    applied: list[AppliedTransform] = []

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if (series < 0).any():
            continue  # sqrt not safe for negative values

        orig_min = round(float(series.min()), 4)
        orig_max = round(float(series.max()), 4)
        df[col] = np.sqrt(df[col])
        applied.append(AppliedTransform(
            column=col,
            transform_type="sqrt_transform",
            original_min=orig_min,
            original_max=orig_max,
            note="Square root transformation applied.",
        ))

    return df, applied


def apply_first_difference(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[AppliedTransform]]:
    """
    Applies first differencing — subtracts each value from the previous.
    Removes one row (first row becomes NaN and is dropped).
    Used for autocorrelation rectification in time-series contexts.
    """
    df = df.copy()
    applied: list[AppliedTransform] = []

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        orig_min = round(float(series.min()), 4)
        orig_max = round(float(series.max()), 4)
        df[col] = df[col].diff()
        applied.append(AppliedTransform(
            column=col,
            transform_type="first_difference",
            original_min=orig_min,
            original_max=orig_max,
            note="First differencing applied. First row dropped (NaN).",
        ))

    df = df.dropna()
    return df, applied


def drop_outliers_iqr(
    df: pd.DataFrame,
    dependent_var: str,
) -> tuple[pd.DataFrame, int]:
    """
    Drops rows where dependent_var is an IQR outlier.
    Returns (cleaned_df, rows_dropped).
    """
    series = df[dependent_var].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    mask = (df[dependent_var] >= q1 - 1.5 * iqr) & (df[dependent_var] <= q3 + 1.5 * iqr)
    cleaned = df[mask]
    return cleaned, int((~mask).sum())


def drop_influential_points(
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Drops influential observations using Cook's distance threshold (4/n).
    Fits OLS, computes Cook's distance, removes points above threshold.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    X = df[independent_vars].dropna()
    y = df[dependent_var].loc[X.index]
    X_const = add_constant(X)

    model = OLS(y, X_const).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    threshold = 4 / len(df)
    influential_mask = cooks_d > threshold
    rows_to_drop = df.index[X.index][influential_mask]
    cleaned = df.drop(index=rows_to_drop)
    return cleaned, len(rows_to_drop)


# ─────────────────────────────────────────────
# MAIN — RESOLVE COLUMN TARGETS FOR TRANSFORMS
# ─────────────────────────────────────────────

def resolve_transform_columns(
    action_details: dict,
    dependent_var: str | None,
    independent_vars: list[str],
) -> list[str]:
    """
    Resolves which columns a transform should be applied to
    based on the action_details 'columns' field.
    """
    target = action_details.get("columns", "dependent")
    if target == "dependent":
        return [dependent_var] if dependent_var else []
    elif target == "independent":
        return independent_vars
    elif target == "all":
        cols = []
        if dependent_var:
            cols.append(dependent_var)
        cols.extend(independent_vars)
        return cols
    return []


# ─────────────────────────────────────────────
# MAIN — APPLY CHOSEN SOLUTION
# ─────────────────────────────────────────────

def apply_solution(
    solution: ProposedSolution,
    df: pd.DataFrame,
    dependent_var: str | None,
    independent_vars: list[str],
) -> tuple[pd.DataFrame, list[AppliedTransform], str | None, str | None]:
    """
    Applies the chosen rectification solution to the DataFrame.

    Returns:
        rectified_df:      DataFrame after applying the solution
        applied_transforms: List of transform logs (empty if not a transform)
        new_test:          New test name if solution is a test switch, else None
        correction_type:   Correction name if solution is a correction, else None
    """
    applied_transforms: list[AppliedTransform] = []
    new_test: str | None = None
    correction_type: str | None = None

    if solution.action_type == "transform":
        fn_name = solution.action_details.get("fn")
        columns = resolve_transform_columns(
            solution.action_details, dependent_var, independent_vars
        )

        if fn_name == "log_transform":
            df, applied_transforms = apply_log_transform(df, columns)
        elif fn_name == "sqrt_transform":
            df, applied_transforms = apply_sqrt_transform(df, columns)
        elif fn_name == "first_difference":
            df, applied_transforms = apply_first_difference(df, columns)

    elif solution.action_type == "test_switch":
        new_test = solution.action_details.get("new_test")

    elif solution.action_type == "correction":
        correction_type = solution.action_details.get("correction_type")

    elif solution.action_type == "drop":
        target = solution.action_details.get("target")
        if target == "outliers" and dependent_var:
            df, dropped = drop_outliers_iqr(df, dependent_var)
        elif target == "influential_points" and dependent_var and independent_vars:
            df, dropped = drop_influential_points(df, dependent_var, independent_vars)
    elif solution.action_type == "drop_variable":
        column_to_drop = solution.action_details.get("column")
        if column_to_drop and column_to_drop in df.columns:
            df = df.drop(columns=[column_to_drop])
            # Remove from independent_vars tracking via transform log
            applied_transforms = [AppliedTransform(
                transform_type="drop_variable",
                column=column_to_drop,
                description=f"Dropped '{column_to_drop}' from the model due to high multicollinearity (VIF ≥ 10).",
                note=f"Re-run assumption checks to verify multicollinearity is resolved.",
            )]
            # Also remove from independent_vars list so downstream engines don't use it
            if column_to_drop in independent_vars:
                independent_vars.remove(column_to_drop)

    return df, applied_transforms, new_test, correction_type


# ─────────────────────────────────────────────
# MAIN — BUILD RectificationOutput
# ─────────────────────────────────────────────

def build_rectification_output(
    failed_assumptions: list[str],
    phase: RectificationPhase,
    chosen_solution_id: str | None,
    df: pd.DataFrame,
    dependent_var: str | None,
    independent_vars: list[str],
    rectification_attempt: int = 1,
    max_attempts: int = 3,
    user_accepted_violation: bool = False,
    accepted_violation_names: list[str] | None = None,
) -> tuple[pd.DataFrame, RectificationOutput]:
    """
    Main entry point for the rectification engine.

    If chosen_solution_id is None → only proposals are generated (first call).
    If chosen_solution_id is set → applies the chosen solution (second call after user picks).
    If user_accepted_violation is True → no solution applied, routes to next step.
    """
    proposals = get_proposals_for_failures(failed_assumptions, phase)

    # ── User accepted violation — no rectification needed ──
    if user_accepted_violation:
        return df, RectificationOutput(
            phase=phase,
            failed_assumptions=failed_assumptions,
            proposed_solutions=proposals,
            next_step=NextStep.STATISTICIAN,
            rectification_attempt=rectification_attempt,
            max_attempts=max_attempts,
            user_accepted_violation=True,
            accepted_violation_names=accepted_violation_names or failed_assumptions,
        )

    # ── No solution chosen yet — return proposals only ──
    if chosen_solution_id is None:
        return df, RectificationOutput(
            phase=phase,
            failed_assumptions=failed_assumptions,
            proposed_solutions=proposals,
            rectification_attempt=rectification_attempt,
            max_attempts=max_attempts,
        )

    # ── Find chosen solution ──
    chosen = next((p for p in proposals if p.solution_id == chosen_solution_id), None)
    if chosen is None:
        # Fallback — solution ID not found, return proposals again
        return df, RectificationOutput(
            phase=phase,
            failed_assumptions=failed_assumptions,
            proposed_solutions=proposals,
            rectification_attempt=rectification_attempt,
            max_attempts=max_attempts,
        )

    # ── Apply chosen solution ──
    rectified_df, applied_transforms, new_test, correction_type = apply_solution(
        solution=chosen,
        df=df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
    )

    # ── Determine next step ──
    next_step_str = chosen.next_step
    try:
        next_step = NextStep(next_step_str)
    except ValueError:
        next_step = NextStep.ASSUMPTION_CHECKER

    rectified_data_json = rectified_df.to_json(orient="records")

    return rectified_df, RectificationOutput(
        phase=phase,
        failed_assumptions=failed_assumptions,
        proposed_solutions=proposals,
        chosen_solution_id=chosen_solution_id,
        chosen_solution=chosen,
        next_step=next_step,
        new_test=new_test,
        applied_transforms=applied_transforms,
        rectified_data_json=rectified_data_json,
        correction_type=correction_type,
        rectification_attempt=rectification_attempt,
        max_attempts=max_attempts,
    )