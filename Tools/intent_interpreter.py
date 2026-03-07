"""
FILE: tools/intent_interpreter_tools.py
-----------------------------------------
LangChain tools exposed to the Intent Interpreter ReAct agent.
The LLM uses these tools to access the profiler output and
parse the user's query into a structured intent object.
"""

import json
from langchain_core.tools import tool

from core.intent_engine import build_intent_output, detect_explicit_test


# ─────────────────────────────────────────────
# SESSION STORE
# Holds profiler output and final intent result
# so tools share state within a single agent run.
# ─────────────────────────────────────────────

_intent_store: dict = {
    "profiler_output": None,    # set before agent is invoked
    "original_query": "",       # set before agent is invoked
    "intent_output": None,      # set after parse_intent_from_llm is called
}


def init_intent_store(profiler_output: dict, original_query: str) -> None:
    """
    Called by run_intent_interpreter() before invoking the agent.
    Seeds the store so tools have access to context.
    """
    _intent_store["profiler_output"] = profiler_output
    _intent_store["original_query"] = original_query
    _intent_store["intent_output"] = None


def get_intent_store() -> dict:
    """Expose store to the agent runner for retrieving final IntentOutput."""
    return _intent_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_dataset_summary() -> str:
    """
    Returns a summary of the dataset from the Data Profiler output.
    Includes column names, their types (continuous/categorical),
    and any warnings flagged during profiling.
    Use this first to understand what columns are available
    before trying to interpret the user's query.
    """
    profiler = _intent_store.get("profiler_output")
    if not profiler:
        return "ERROR: No profiler output available. Cannot interpret intent without dataset info."

    continuous = [c["column"] for c in profiler.get("continuous_columns", [])]
    categorical = [c["column"] for c in profiler.get("categorical_columns", [])]
    warnings = profiler.get("warnings", [])
    fatal = profiler.get("fatal_errors", [])

    summary = {
        "n_rows": profiler.get("n_rows"),
        "n_cols": profiler.get("n_cols"),
        "continuous_columns": continuous,
        "categorical_columns": categorical,
        "profiler_warnings": warnings,
        "fatal_errors": fatal,
    }
    return json.dumps(summary, indent=2)


@tool
def get_user_query() -> str:
    """
    Returns the original raw query submitted by the user.
    Use this to understand what the user is asking for.
    """
    query = _intent_store.get("original_query", "")
    if not query:
        return "ERROR: No user query found in store."
    return query


@tool
def check_explicit_test_in_query() -> str:
    """
    Runs a deterministic scan of the user's query to check if they
    explicitly named a known statistical test (e.g. 't-test', 'ANOVA', 'PCA').
    Returns the canonical test name if found, or 'None' if not detected.
    Use this early — if a test is detected, set intent_type to 'explicit_test'
    and set methodologist_bypass to true.
    """
    query = _intent_store.get("original_query", "")
    result = detect_explicit_test(query)
    return result if result else "None"

@tool
def parse_intent_from_llm(llm_parsed_json: str) -> str:
    """
    Validates your structured interpretation against the actual dataset columns.

    CRITICAL — HOW TO CALL THIS TOOL:
    You MUST pass a single argument named exactly 'llm_parsed_json' whose value
    is your entire JSON as a STRING. Do NOT pass JSON keys as separate arguments.

    Correct:   parse_intent_from_llm(llm_parsed_json='{\"intent_type\": ...}')
    Wrong:     parse_intent_from_llm(intent_type="...", columns=[...])

    Required keys in the JSON string:
        intent_type            : "explicit_test" | "column_relationship" | "open_ended"
        analysis_goal          : "prediction" | "inference" | "dimensionality" |
                                 "relationship" | "distribution" | "unknown"
        confidence             : "high" | "medium" | "low"
        requested_test         : string or null
        columns                : list of {"name": str, "role": "dependent"|"independent"|"grouping"|"unspecified"}
        all_columns_mode       : true | false
        interpretation_summary : plain English summary of what you understood
        clarification_needed   : false
        clarification_question : null

    Returns the final validated IntentOutput as a JSON string.
    """
    profiler = _intent_store.get("profiler_output")
    query = _intent_store.get("original_query", "")

    if not profiler:
        return "ERROR: No profiler output in store. Cannot validate columns."

    try:
        llm_parsed = json.loads(llm_parsed_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON provided — {str(e)}"

    try:
        intent_output = build_intent_output(
            llm_parsed=llm_parsed,
            profiler_output=profiler,
            original_query=query,
        )
        _intent_store["intent_output"] = intent_output
        return intent_output.model_dump_json(indent=2)
    except ValueError as e:
        return f"FATAL_COLUMN_ERROR: {str(e)}"
    except Exception as e:
        return f"ERROR: Failed to build IntentOutput — {str(e)}"


@tool
def get_column_details(column_name: str) -> str:
    """
    Returns detailed profiler stats for a specific column.
    Useful when you need to know more about a column's type,
    skewness, or missing data before deciding its role in the analysis.
    Args:
        column_name: The exact name of the column to look up.
    """
    profiler = _intent_store.get("profiler_output")
    if not profiler:
        return "ERROR: No profiler output available."

    for col in profiler.get("continuous_columns", []):
        if col["column"] == column_name:
            return json.dumps({"type": "continuous", **col}, indent=2)

    for col in profiler.get("categorical_columns", []):
        if col["column"] == column_name:
            return json.dumps({"type": "categorical", **col}, indent=2)

    return f"ERROR: Column '{column_name}' not found in profiler output."


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

INTENT_INTERPRETER_TOOLS = [
    get_dataset_summary,
    get_user_query,
    check_explicit_test_in_query,
    parse_intent_from_llm,
    get_column_details,
]