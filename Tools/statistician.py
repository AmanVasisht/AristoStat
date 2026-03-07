"""
FILE: tools/statistician_tools.py
------------------------------------
LangChain tools exposed to the Statistician ReAct agent.
The fitted model object is stored in memory here so the
Model Critic can access it directly without serialization.
"""

import json
import pandas as pd
from langchain_core.tools import tool

from core.statistician_engine import run_test


# ─────────────────────────────────────────────
# SESSION STORE
# _fitted_model is kept separate from the serializable store
# so it can be passed directly to the Model Critic in memory.
# ─────────────────────────────────────────────

_statistician_store: dict = {
    "methodologist_output":  None,
    "rectification_output":  None,   # may be None if no rectification was needed
    "cleaned_df":            None,
    "statistician_output":   None,
}

_fitted_model = None   # stored at module level — not serialized


def init_statistician_store(
    methodologist_output: dict,
    cleaned_df: pd.DataFrame,
    rectification_output: dict | None = None,
) -> None:
    """
    Called by run_statistician() before invoking the agent.
    rectification_output is optional — None if no failures occurred.
    """
    global _fitted_model
    _statistician_store["methodologist_output"] = methodologist_output
    _statistician_store["rectification_output"] = rectification_output
    _statistician_store["cleaned_df"]           = cleaned_df
    _statistician_store["statistician_output"]  = None
    _fitted_model = None


def get_statistician_store() -> dict:
    """Expose store to the agent runner."""
    return _statistician_store


def get_fitted_model() -> object | None:
    """
    Returns the fitted model object stored in memory.
    Called directly by run_model_critic() in main.py to pass
    the model to the Model Critic without serialization.
    """
    return _fitted_model


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_test_context() -> str:
    """
    Returns the test to run, column roles, and any correction to apply.
    Reads correction_type from RectificationOutput automatically if present.
    Call this first to understand exactly what test to run and with which columns.
    """
    meth  = _statistician_store.get("methodologist_output", {})
    rect  = _statistician_store.get("rectification_output") or {}

    correction_type = rect.get("correction_type")
    new_test        = rect.get("new_test")

    # If rectification switched the test, use the new test name
    test_name = new_test or meth.get("selected_test")

    return json.dumps({
        "test_name":            test_name,
        "dependent_variable":   meth.get("dependent_variable"),
        "independent_variables": meth.get("independent_variables", []),
        "grouping_variable":    meth.get("grouping_variable"),
        "correction_type":      correction_type,
        "n_rows":               meth.get("n_rows"),
        "rectification_applied": bool(rect),
        "user_accepted_violations": rect.get("accepted_violation_names", []),
    }, indent=2)


@tool
def execute_test() -> str:
    """
    Executes the statistical test using the context from get_test_context().
    Reads test name, column roles, and correction type automatically from store.
    For regression tests, the fitted model is stored in memory for the Model Critic.

    Returns the complete StatisticianOutput as a JSON string.
    Check the model_available_in_memory field — if True, the Model Critic
    should be called next to run post-test assumption checks.
    """
    global _fitted_model

    meth = _statistician_store.get("methodologist_output", {})
    rect = _statistician_store.get("rectification_output") or {}
    df   = _statistician_store.get("cleaned_df")

    if df is None:
        return "ERROR: No DataFrame in store."

    correction_type = rect.get("correction_type")
    new_test        = rect.get("new_test")
    test_name       = new_test or meth.get("selected_test")
    dep_var         = meth.get("dependent_variable")
    ind_vars        = meth.get("independent_variables", [])
    grp_var         = meth.get("grouping_variable")

    if not test_name:
        return "ERROR: No test name available."

    try:
        output, model = run_test(
            test_name=test_name,
            df=df,
            dependent_var=dep_var,
            independent_vars=ind_vars,
            grouping_var=grp_var,
            correction_type=correction_type,
        )

        # Store fitted model in memory for Model Critic
        _fitted_model = model
        _statistician_store["statistician_output"] = output

        return output.model_dump_json(indent=2)

    except Exception as e:
        return f"ERROR: Test execution failed — {str(e)}"


@tool
def get_result_summary() -> str:
    """
    Returns a simplified summary of the test results — just the key numbers
    and interpretation. Use this after execute_test to prepare what to
    present to the user clearly.
    """
    output = _statistician_store.get("statistician_output")
    if output is None:
        return "ERROR: No results available. Run execute_test first."

    summary = {
        "test_name":    output.test_name,
        "test_family":  output.test_family,
        "n_observations": output.n_observations,
        "correction_applied": output.correction_applied,
    }

    if output.inference_result:
        r = output.inference_result
        summary["statistic"]    = f"{r.statistic_label}={r.statistic}"
        summary["p_value"]      = r.p_value
        summary["verdict"]      = r.verdict
        summary["effect_size"]  = f"{r.effect_size_label}={r.effect_size}" if r.effect_size else None
        summary["interpretation"] = r.interpretation

    elif output.regression_result:
        r = output.regression_result
        summary["r_squared"]     = r.r_squared
        summary["adj_r_squared"] = r.adj_r_squared
        summary["f_statistic"]   = r.f_statistic
        summary["f_p_value"]     = r.f_p_value
        summary["interpretation"] = r.interpretation

    elif output.correlation_result:
        r = output.correlation_result
        summary["statistic"]          = f"{r.statistic_label}={r.statistic}"
        summary["p_value"]            = r.p_value
        summary["verdict"]            = r.verdict
        summary["correlation_strength"] = r.correlation_strength
        summary["interpretation"]     = r.interpretation

    elif output.dimensionality_result:
        r = output.dimensionality_result
        summary["n_components_selected"]   = r.n_components_selected
        summary["total_variance_explained"] = r.total_variance_explained
        summary["interpretation"]          = r.interpretation

    return json.dumps(summary, indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

STATISTICIAN_TOOLS = [
    get_test_context,
    execute_test,
    get_result_summary,
]