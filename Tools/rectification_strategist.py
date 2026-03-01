"""
FILE: tools/rectification_tools.py
------------------------------------
LangChain tools exposed to the Rectification Strategist ReAct agent.
Used for both pre-test (from Assumption Checker) and
post-test (from Model Critic) failures — distinguished by phase flag.
"""

import json
import pandas as pd
from langchain_core.tools import tool

from Schemas.rectification_strategist import RectificationPhase
from core.rectification_engine import build_rectification_output, get_proposals_for_failures


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_rectification_store: dict = {
    "failed_assumptions":   None,   # list[str] — assumption names that failed
    "phase":                None,   # "pre_test" | "post_test"
    "cleaned_df":           None,   # pd.DataFrame — current working data
    "dependent_var":        None,   # str | None
    "independent_vars":     None,   # list[str]
    "methodologist_output": None,   # for column role reference
    "rectification_attempt": 1,
    "max_attempts":          3,
    "rectification_output": None,   # RectificationOutput — set after apply_chosen_solution
}


def init_rectification_store(
    failed_assumptions: list[str],
    phase: str,
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
    rectification_attempt: int = 1,
    max_attempts: int = 3,
) -> None:
    """Called by run_rectification_strategist() before invoking the agent."""
    _rectification_store["failed_assumptions"]    = failed_assumptions
    _rectification_store["phase"]                 = phase
    _rectification_store["cleaned_df"]            = cleaned_df
    _rectification_store["dependent_var"]         = methodologist_output.get("dependent_variable")
    _rectification_store["independent_vars"]      = methodologist_output.get("independent_variables", [])
    _rectification_store["methodologist_output"]  = methodologist_output
    _rectification_store["rectification_attempt"] = rectification_attempt
    _rectification_store["max_attempts"]          = max_attempts
    _rectification_store["rectification_output"]  = None


def get_rectification_store() -> dict:
    """Expose store to the agent runner."""
    return _rectification_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_failure_context() -> str:
    """
    Returns context about the current failure situation:
    which assumptions failed, the phase (pre_test or post_test),
    current rectification attempt number, and max allowed attempts.
    Call this first to understand what needs to be rectified.
    """
    store = _rectification_store
    return json.dumps({
        "failed_assumptions":    store.get("failed_assumptions", []),
        "phase":                 store.get("phase"),
        "rectification_attempt": store.get("rectification_attempt"),
        "max_attempts":          store.get("max_attempts"),
        "dependent_variable":    store.get("dependent_var"),
        "independent_variables": store.get("independent_vars", []),
    }, indent=2)


@tool
def get_proposed_solutions() -> str:
    """
    Returns the list of proposed solutions for the failed assumptions,
    filtered by the current phase (pre_test or post_test).
    Solutions are ranked: corrections first, then test switches, transforms, drops.
    Present these options to the user and ask them to choose one.
    """
    failed = _rectification_store.get("failed_assumptions", [])
    phase_str = _rectification_store.get("phase", "pre_test")

    try:
        phase = RectificationPhase(phase_str)
    except ValueError:
        phase = RectificationPhase.PRE_TEST

    proposals = get_proposals_for_failures(failed, phase)

    if not proposals:
        return json.dumps({
            "message": "No automated solutions available for these failures.",
            "suggestions": "Consider collecting more data, consulting a statistician, "
                           "or proceeding with the test while acknowledging the violations."
        }, indent=2)

    # ── Save proposals to store so orchestrator can access them ──
    from Schemas.rectification_strategist import RectificationOutput
    output = RectificationOutput(
        phase=phase,
        failed_assumptions=failed,
        proposed_solutions=proposals,
    )
    _rectification_store["rectification_output"] = output

    return json.dumps([
        {
            "solution_id":  p.solution_id,
            "description":  p.description,
            "action_type":  p.action_type,
            "next_step":    p.next_step,
        }
        for p in proposals
    ], indent=2)


@tool
def apply_chosen_solution(solution_id: str) -> str:
    """
    Applies the user's chosen rectification solution.
    For transforms: modifies the DataFrame in memory.
    For test switches: records the new test name.
    For corrections: records the correction type.

    Returns the RectificationOutput as a JSON string.
    Check the following fields:
      - next_step:         where to route after this ("assumption_checker" or "statistician")
      - new_test:          set if solution was a test switch
      - correction_type:   set if solution was a statistical correction
      - applied_transforms: list of data changes made

    Args:
        solution_id: The solution_id chosen by the user from get_proposed_solutions().
    """
    failed       = _rectification_store.get("failed_assumptions", [])
    phase_str    = _rectification_store.get("phase", "pre_test")
    df           = _rectification_store.get("cleaned_df")
    dep_var      = _rectification_store.get("dependent_var")
    ind_vars     = _rectification_store.get("independent_vars", [])
    attempt      = _rectification_store.get("rectification_attempt", 1)
    max_attempts = _rectification_store.get("max_attempts", 3)

    if df is None:
        return "ERROR: No DataFrame in store."

    try:
        phase = RectificationPhase(phase_str)
    except ValueError:
        phase = RectificationPhase.PRE_TEST

    try:
        rectified_df, output = build_rectification_output(
            failed_assumptions=failed,
            phase=phase,
            chosen_solution_id=solution_id,
            df=df,
            dependent_var=dep_var,
            independent_vars=ind_vars,
            rectification_attempt=attempt,
            max_attempts=max_attempts,
        )
        # Update store with rectified data
        _rectification_store["cleaned_df"]          = rectified_df
        _rectification_store["rectification_output"] = output
        return output.model_dump_json(indent=2)
    except Exception as e:
        return f"ERROR: Failed to apply solution — {str(e)}"


@tool
def accept_violation_and_proceed(violation_names: str) -> str:
    """
    Called when the user chooses to proceed despite assumption failures
    (instead of rectifying). Records the accepted violations and routes
    directly to the Statistician.

    Args:
        violation_names: Comma-separated list of assumption names the user is accepting.
                         Example: "normality,homogeneity_of_variance"
    """
    failed       = _rectification_store.get("failed_assumptions", [])
    phase_str    = _rectification_store.get("phase", "pre_test")
    df           = _rectification_store.get("cleaned_df")
    dep_var      = _rectification_store.get("dependent_var")
    ind_vars     = _rectification_store.get("independent_vars", [])
    attempt      = _rectification_store.get("rectification_attempt", 1)
    max_attempts = _rectification_store.get("max_attempts", 3)

    accepted = [v.strip() for v in violation_names.split(",")]

    try:
        phase = RectificationPhase(phase_str)
    except ValueError:
        phase = RectificationPhase.PRE_TEST

    _, output = build_rectification_output(
        failed_assumptions=failed,
        phase=phase,
        chosen_solution_id=None,
        df=df,
        dependent_var=dep_var,
        independent_vars=ind_vars,
        rectification_attempt=attempt,
        max_attempts=max_attempts,
        user_accepted_violation=True,
        accepted_violation_names=accepted,
    )
    _rectification_store["rectification_output"] = output
    return output.model_dump_json(indent=2)


@tool
def check_attempt_limit() -> str:
    """
    Returns whether the maximum rectification attempt limit has been reached.
    If max attempts exceeded, the user must either accept violations or stop.
    """
    attempt      = _rectification_store.get("rectification_attempt", 1)
    max_attempts = _rectification_store.get("max_attempts", 3)
    exceeded     = attempt > max_attempts

    return json.dumps({
        "current_attempt": attempt,
        "max_attempts":    max_attempts,
        "limit_exceeded":  exceeded,
        "message": (
            f"Maximum rectification attempts ({max_attempts}) reached. "
            f"Please either accept the remaining violations or stop the analysis."
            if exceeded else
            f"Attempt {attempt} of {max_attempts}."
        )
    }, indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

RECTIFICATION_TOOLS = [
    get_failure_context,
    get_proposed_solutions,
    apply_chosen_solution,
    accept_violation_and_proceed,
    check_attempt_limit,
]