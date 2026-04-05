"""
FILE: agents/statistician.py
------------------------------
Statistician agent — wires together all components and exposes
the public run_statistician() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/statistician_prompt.py
  - Tools   ← tools/statistician_tools.py
  - Engine  ← core/statistician_engine.py  (called via tools)
  - Schemas ← schemas/statistician_schema.py

Note on fitted model passing:
  The fitted model (statsmodels OLS, sklearn Pipeline, etc.) is stored
  at module level in tools/statistician_tools.py via get_fitted_model().
  The orchestrator calls get_fitted_model() and passes it directly to
  run_model_critic() — no serialization needed.
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langsmith import traceable, get_current_run_tree

from Tools.statistician import (
    STATISTICIAN_TOOLS,
    init_statistician_store,
    get_statistician_store,
    get_fitted_model,
    _statistician_store,
)


# ─────────────────────────────────────────────
# LLM — used only for formatting the response
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

@traceable(
    name="statistician",
    tags=["agent", "direct-call", "test-execution"],
    metadata={"agent_pattern": "direct", "model": "llama-3.1-8b-instant"}
)
def run_statistician(
    methodologist_output: dict,
    cleaned_df: pd.DataFrame,
    rectification_output: dict | None = None,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    Calls the statistician engine directly — no LLM needed for test execution.
    The LLM is only used to format the final human-readable response.
    """
    from core.statistician_engine import run_test

    init_statistician_store(
        methodologist_output=methodologist_output,
        cleaned_df=cleaned_df,
        rectification_output=rectification_output,
    )

    # ── Resolve test parameters ──
    rect       = rectification_output or {}
    test_name  = rect.get("new_test") or methodologist_output.get("selected_test")
    dep_var    = methodologist_output.get("dependent_variable")
    ind_vars   = methodologist_output.get("independent_variables", [])
    grp_var    = methodologist_output.get("grouping_variable")
    correction = rect.get("correction_type")

    # ── Execute test directly via engine ──
    statistician_output = None
    fitted_model = None
    final_response = ""
    test_failed = False
    error_message = None

    try:
        output, fitted_model = run_test(
            test_name=test_name,
            df=cleaned_df,
            dependent_var=dep_var,
            independent_vars=ind_vars,
            grouping_var=grp_var,
            correction_type=correction,
        )
        _statistician_store["statistician_output"] = output
        import Tools.statistician as _st
        _st._fitted_model = fitted_model
        statistician_output = output

        result_json = output.model_dump_json(indent=2)
        format_prompt = f"""You are presenting statistical results clearly.
    Given these results, write a clear 3-5 sentence summary for the user.
    Include: test name, key statistic, p-value, verdict, and brief plain English interpretation.
    Do not add caveats or recommendations — just present the results.

    Results:
    {result_json}"""

        response = model.invoke([HumanMessage(content=format_prompt)])
        final_response = response.content.strip()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        final_response = f"Test execution failed: {str(e)}"
        test_failed = True
        error_message = str(e)

    statistician_output_dict = (
        statistician_output.model_dump() if statistician_output else {}
    )

    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        p_value = statistician_output_dict.get("p_value")
        
        run.add_metadata({
            # Test identity — was rectification applied?
            "test_executed": test_name,
            "original_test": methodologist_output.get("selected_test"),
            "rectification_applied": bool(rect.get("new_test")),
            "correction_type": correction,

            # Core results — the most queryable fields across runs
            "p_value": p_value,
            "significant": p_value < 0.05 if p_value is not None else None,
            "test_statistic": statistician_output_dict.get("statistic"),
            "effect_size": statistician_output_dict.get("effect_size"),

            # Regression-specific — only populated for regression tests
            "r_squared": statistician_output_dict.get("r_squared"),
            "adj_r_squared": statistician_output_dict.get("adj_r_squared"),

            # Execution health
            "test_failed": test_failed,
            "error_message": error_message,
        })

    return {
        "messages":            [],
        "final_response":      final_response,
        "statistician_output": statistician_output_dict,
        "fitted_model":        fitted_model,
    }