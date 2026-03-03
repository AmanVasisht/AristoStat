"""
FILE: core/methodologist_engine.py
------------------------------------
Pure deterministic logic for the Methodologist agent.
No LangChain or LLM dependencies.

Bypass validation rules:
  - If the user named a test and it is clearly wrong (fundamental type mismatch
    or group count impossibility) → OVERRIDE: run decision tree, inform user.
  - If the user named a test and there is a minor issue (e.g. small sample size
    warning, slight imbalance) → WARN: proceed with their test, show warning.
  - If the user named a test and everything looks fine → BYPASS: pass straight through.

"Clearly wrong" criteria (triggers override):
  1. Dependent variable is categorical but test requires continuous.
  2. Test requires a grouping variable but none exists in the columns.
  3. Test requires exactly 2 groups but grouping column has 3+ levels (or vice versa).
  4. Test requires continuous predictors but all independents are categorical.
"""

from itertools import combinations as _combinations

from Schemas.methodologist import MethodologistOutput, SelectionMode
from Schemas.intent_interpreter import IntentOutput
from Utils.test_requirements_registry import TEST_REQUIREMENTS

# ─────────────────────────────────────────────
# HELPER — BUILD DTYPE MAP FROM PROFILER
# ─────────────────────────────────────────────

def _build_dtype_map(profiler_output: dict) -> dict[str, str]:
    col_map: dict[str, str] = {}
    for col in profiler_output.get("continuous_columns", []):
        col_map[col["column"]] = "continuous"
    for col in profiler_output.get("categorical_columns", []):
        col_map[col["column"]] = "categorical"
    return col_map


def _get_dtype(col_name: str | None, profiler_output: dict) -> str | None:
    if not col_name:
        return None
    return _build_dtype_map(profiler_output).get(col_name, "unknown")


# ─────────────────────────────────────────────
# HELPER — EXTRACT COLUMN ROLES
# ─────────────────────────────────────────────

def extract_column_roles(
    intent: IntentOutput,
) -> tuple[str | None, list[str], str | None]:
    """
    Extracts dependent, independent, and grouping column names
    from the ColumnReference list in IntentOutput.
    Unspecified roles are inferred from dtype — categorical → grouping,
    continuous → independent.
    """
    dependent_var: str | None = None
    independent_vars: list[str] = []
    grouping_var: str | None = None

    for col in intent.columns:
        if col.role == "dependent":
            dependent_var = col.name
        elif col.role == "independent":
            independent_vars.append(col.name)
        elif col.role == "grouping":
            grouping_var = col.name
        else:
            if col.dtype_from_profiler == "categorical":
                grouping_var = col.name
            else:
                independent_vars.append(col.name)

    return dependent_var, independent_vars, grouping_var


# ─────────────────────────────────────────────
# HELPER — GET N_GROUPS FROM PROFILER
# ─────────────────────────────────────────────

def get_n_groups(grouping_var: str | None, profiler_output: dict) -> int | None:
    if not grouping_var:
        return None
    for col in profiler_output.get("categorical_columns", []):
        if col["column"] == grouping_var:
            return col.get("cardinality")
    return None


# ─────────────────────────────────────────────
# BYPASS VALIDATION
# Returns (is_clearly_wrong, override_reason, mismatch_warning)
#
# is_clearly_wrong=True  → OVERRIDE: run decision tree instead
# is_clearly_wrong=False → WARN or BYPASS depending on mismatch_warning
# ─────────────────────────────────────────────

def validate_bypass(
    test_name: str,
    dependent_var: str | None,
    independent_vars: list[str],
    grouping_var: str | None,
    n_groups: int | None,
    profiler_output: dict,
) -> tuple[bool, str | None, str | None]:
    """
    Validates whether the user-named test is appropriate for the given columns.

    Returns:
        (is_clearly_wrong, override_reason, mismatch_warning)

        is_clearly_wrong=True  + override_reason set → hard override, use decision tree
        is_clearly_wrong=False + mismatch_warning set → soft warning, proceed with user test
        is_clearly_wrong=False + mismatch_warning None → clean bypass
    """
    if test_name not in TEST_REQUIREMENTS:
        # Unknown test — cannot validate, clean bypass
        return False, None, None

    req = TEST_REQUIREMENTS[test_name]
    dtype_map = _build_dtype_map(profiler_output)

    dep_dtype = dtype_map.get(dependent_var, "unknown") if dependent_var else None
    ind_dtypes = [dtype_map.get(iv, "unknown") for iv in independent_vars]

    hard_errors: list[str] = []    # clearly wrong → override
    soft_warnings: list[str] = []  # minor issue → warn but proceed

    # ── HARD CHECK 1: Dependent variable type ──
    if req["dependent"] and req["dependent"] != "any" and dep_dtype:
        if dep_dtype != req["dependent"]:
            hard_errors.append(
                f"'{test_name}' requires a {req['dependent']} dependent variable, "
                f"but '{dependent_var}' is {dep_dtype}. "
                f"This is a fundamental mismatch."
            )

    # ── HARD CHECK 2: Grouping variable required but missing ──
    if req["grouping"] is True and not grouping_var:
        hard_errors.append(
            f"'{test_name}' requires a categorical grouping variable "
            f"(to define comparison groups), but none was identified in your columns."
        )

    # ── HARD CHECK 3: Grouping variable present but test doesn't use one ──
    if req["grouping"] is False and grouping_var:
        hard_errors.append(
            f"'{test_name}' does not use a grouping variable, "
            f"but '{grouping_var}' was provided as a grouping column. "
            f"This suggests a different test may be intended."
        )

    # ── HARD CHECK 4: Group count mismatch ──
    if n_groups is not None:
        if req["max_groups"] and n_groups > req["max_groups"]:
            hard_errors.append(
                f"'{test_name}' supports at most {req['max_groups']} group(s), "
                f"but '{grouping_var}' has {n_groups} unique levels. "
                f"This is incompatible."
            )
        if req["min_groups"] and n_groups < req["min_groups"]:
            hard_errors.append(
                f"'{test_name}' requires at least {req['min_groups']} group(s), "
                f"but '{grouping_var}' has only {n_groups} unique level(s). "
                f"This is incompatible."
            )

    # ── HARD CHECK 5: Independent variable types ──
    if req["independent"] and req["independent"] != "any" and ind_dtypes:
        wrong_ind = [
            iv for iv, dt in zip(independent_vars, ind_dtypes)
            if dt != req["independent"]
        ]
        if wrong_ind:
            hard_errors.append(
                f"'{test_name}' expects {req['independent']} independent variable(s), "
                f"but {wrong_ind} are not {req['independent']}."
            )

    # ── SOFT CHECK: Sample size ──
    n_rows = profiler_output.get("n_rows", 0)
    if req["min_n"] and n_rows < req["min_n"]:
        soft_warnings.append(
            f"Sample size (n={n_rows}) is below the recommended minimum of "
            f"{req['min_n']} for '{test_name}'. Results should be interpreted cautiously."
        )

    if hard_errors:
        return True, " | ".join(hard_errors), None

    if soft_warnings:
        return False, None, " | ".join(soft_warnings)

    return False, None, None


# ─────────────────────────────────────────────
# DECISION TREE
# ─────────────────────────────────────────────

def decide_test(
    intent: IntentOutput,
    profiler_output: dict,
    dependent_var: str | None,
    independent_vars: list[str],
    grouping_var: str | None,
    n_groups: int | None,
    n_rows: int,
) -> tuple[str, str]:
    """
    Deterministic decision tree that selects the most appropriate test.
    Returns (selected_test_name, reasoning_string).
    """
    dtype_map = _build_dtype_map(profiler_output)
    dep_dtype  = dtype_map.get(dependent_var, "unknown") if dependent_var else None
    grp_dtype  = dtype_map.get(grouping_var, "unknown")  if grouping_var  else None
    ind_dtypes = [dtype_map.get(iv, "unknown") for iv in independent_vars]
    goal = intent.analysis_goal.value

    # ── DIMENSIONALITY ──
    if goal == "dimensionality":
        return (
            "Principal Component Analysis",
            f"PCA selected — your goal is dimensionality reduction across "
            f"{len(independent_vars)} continuous variable(s)."
        )

    # ── RELATIONSHIP / CORRELATION ──
    if goal in ("relationship", "prediction"):
        if dep_dtype == "continuous" and all(d == "continuous" for d in ind_dtypes) and not grouping_var:
            if len(independent_vars) == 1:
                return (
                    "Simple Linear Regression",
                    f"Simple Linear Regression selected — one continuous outcome "
                    f"('{dependent_var}') and one continuous predictor "
                    f"('{independent_vars[0]}')."
                )
            elif len(independent_vars) > 1:
                return (
                    "Multiple Linear Regression",
                    f"Multiple Linear Regression selected — predicting '{dependent_var}' "
                    f"from {len(independent_vars)} continuous predictors."
                )
            # Both continuous, no clear direction
            return (
                "Pearson Correlation",
                f"Pearson Correlation selected — both '{dependent_var}' and "
                f"'{independent_vars[0]}' are continuous with no grouping variable."
            )
    # ── PREDICTION ──
    if goal == "prediction":
        if len(independent_vars) == 1:
            return (
                "Simple Linear Regression",
                f"Simple Linear Regression selected — predicting '{dependent_var}' "
                f"from one continuous predictor ('{independent_vars[0]}')."
            )
        elif len(independent_vars) > 1:
            return (
                "Multiple Linear Regression",
                f"Multiple Linear Regression selected — predicting '{dependent_var}' "
                f"from {len(independent_vars)} predictors."
            )
    # ── FACTORIAL ANOVA (Two or More Categorical Predictors) ──
    if (
        goal == "inference"
        and dep_dtype == "continuous"
        and len(independent_vars) >= 2
        and all(d == "categorical" for d in ind_dtypes)
    ):
        if len(independent_vars) == 2:
            return (
                "Two-Way ANOVA",
                f"Two-Way ANOVA selected — analyzing the effect of "
                f"'{independent_vars[0]}' and '{independent_vars[1]}' "
                f"on '{dependent_var}'."
            )
        else:
            return (
                "Factorial ANOVA",
                f"Factorial ANOVA selected — analyzing the effects of "
                f"{len(independent_vars)} categorical predictors "
                f"on '{dependent_var}'."
            )
    # ── INFERENCE ──
    if (goal == "inference" or (dep_dtype == "continuous" and grp_dtype == "categorical")) and grouping_var:
        if n_groups == 2:
            if n_rows >= 30:
                return (
                    "Independent Samples T-Test",
                    f"Independent Samples T-Test selected — comparing '{dependent_var}' "
                    f"across 2 groups in '{grouping_var}' (n={n_rows})."
                )
            else:
                return (
                    "Mann-Whitney U Test",
                    f"Mann-Whitney U Test selected — comparing '{dependent_var}' "
                    f"across 2 groups in '{grouping_var}' (n={n_rows}, small sample)."
                )
        elif n_groups and n_groups >= 3:
            if n_rows >= 20:
                return (
                    "One-Way ANOVA",
                    f"One-Way ANOVA selected — comparing '{dependent_var}' "
                    f"across {n_groups} groups in '{grouping_var}'."
                )
            else:
                return (
                    "Kruskal-Wallis Test",
                    f"Kruskal-Wallis Test selected — comparing '{dependent_var}' "
                    f"across {n_groups} groups in '{grouping_var}' (small sample)."
                )
        # Has grouping var but n_groups unclear — default inference test
        return (
            "Independent Samples T-Test",
            f"Independent Samples T-Test selected as default inference test for '{dependent_var}'."
        )

    # ── FALLBACK: Two continuous columns ──
    if dep_dtype == "continuous" and all(d == "continuous" for d in ind_dtypes):
        return (
            "Pearson Correlation",
            f"Pearson Correlation selected — both columns are continuous "
            f"and no specific goal was stated."
        )

    # ── FINAL FALLBACK ──
    return (
        "Pearson Correlation",
        "Could not determine a specific test from the given inputs. "
        "Defaulting to Pearson Correlation — please verify this is appropriate."
    )


# ─────────────────────────────────────────────
# MAIN — BUILD MethodologistOutput
# ─────────────────────────────────────────────

def build_methodologist_output(
    intent: IntentOutput,
    profiler_output: dict,
) -> MethodologistOutput:
    """
    Main entry point. Handles bypass (with override or warn logic)
    and decision tree paths.
    """
    n_rows = profiler_output.get("n_rows", 0)

    dependent_var, independent_vars, grouping_var = extract_column_roles(intent)
    n_groups = get_n_groups(grouping_var, profiler_output)

    # ── BYPASS PATH ──
    if intent.methodologist_bypass and intent.requested_test:
        is_clearly_wrong, override_reason, mismatch_warning = validate_bypass(
            test_name=intent.requested_test,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            grouping_var=grouping_var,
            n_groups=n_groups,
            profiler_output=profiler_output,
        )

        # ── OVERRIDE: clearly wrong — run decision tree instead ──
        if is_clearly_wrong:
            correct_test, reasoning = decide_test(
                intent=intent,
                profiler_output=profiler_output,
                dependent_var=dependent_var,
                independent_vars=independent_vars,
                grouping_var=grouping_var,
                n_groups=n_groups,
                n_rows=n_rows,
            )
            return MethodologistOutput(
                selected_test=correct_test,
                selection_mode=SelectionMode.OVERRIDDEN,
                user_requested_test=intent.requested_test,
                override_reason=override_reason,
                dependent_variable=dependent_var,
                independent_variables=independent_vars,
                grouping_variable=grouping_var,
                n_rows=n_rows,
                n_groups=n_groups,
                dependent_dtype=_get_dtype(dependent_var, profiler_output),
                independent_dtypes=[_get_dtype(iv, profiler_output) for iv in independent_vars],
                reasoning=reasoning,
                original_query=intent.original_query,
            )

        # ── WARN: minor issue — proceed with user's test ──
        if mismatch_warning:
            return MethodologistOutput(
                selected_test=intent.requested_test,
                selection_mode=SelectionMode.WARNED,
                mismatch_warning=mismatch_warning,
                dependent_variable=dependent_var,
                independent_variables=independent_vars,
                grouping_variable=grouping_var,
                n_rows=n_rows,
                n_groups=n_groups,
                dependent_dtype=_get_dtype(dependent_var, profiler_output),
                independent_dtypes=[_get_dtype(iv, profiler_output) for iv in independent_vars],
                reasoning=(
                    f"You requested '{intent.requested_test}'. "
                    f"Column types are broadly compatible — proceeding with your choice."
                ),
                original_query=intent.original_query,
            )

        # ── CLEAN BYPASS ──
        return MethodologistOutput(
            selected_test=intent.requested_test,
            selection_mode=SelectionMode.BYPASS,
            dependent_variable=dependent_var,
            independent_variables=independent_vars,
            grouping_variable=grouping_var,
            n_rows=n_rows,
            n_groups=n_groups,
            dependent_dtype=_get_dtype(dependent_var, profiler_output),
            independent_dtypes=[_get_dtype(iv, profiler_output) for iv in independent_vars],
            reasoning=f"You explicitly requested '{intent.requested_test}' and column types are compatible.",
            original_query=intent.original_query,
        )

    # ── DECISION TREE PATH ──
    selected_test, reasoning = decide_test(
        intent=intent,
        profiler_output=profiler_output,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        grouping_var=grouping_var,
        n_groups=n_groups,
        n_rows=n_rows,
    )

    return MethodologistOutput(
        selected_test=selected_test,
        selection_mode=SelectionMode.DECIDED,
        dependent_variable=dependent_var,
        independent_variables=independent_vars,
        grouping_variable=grouping_var,
        n_rows=n_rows,
        n_groups=n_groups,
        dependent_dtype=_get_dtype(dependent_var, profiler_output),
        independent_dtypes=[_get_dtype(iv, profiler_output) for iv in independent_vars],
        reasoning=reasoning,
        original_query=intent.original_query,
    )