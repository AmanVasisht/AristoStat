"""
FILE: Agents/critic.py
------------------------------
Model Critic agent — direct engine call, no ReAct agent.
LLM used only for formatting the final response.
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langsmith import traceable, get_current_run_tree

from core.critic_engine import run_post_test_checks


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
    name="model_critic",
    tags=["agent", "direct-call", "post-test-diagnostics"],
    metadata={"agent_pattern": "direct", "model": "llama-3.1-8b-instant"}
)
def run_model_critic(
    statistician_output: dict,
    fitted_model: object | None,
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    Unpacks the output dicts and calls run_post_test_checks directly.
    LLM used only to format the human-readable response.
    """

    # ── Unpack fields that run_post_test_checks expects ──
    test_family      = statistician_output.get("test_family", "")
    test_name        = statistician_output.get("test_name", "")
    dependent_var    = methodologist_output.get("dependent_variable")
    independent_vars = methodologist_output.get("independent_variables", [])

    critic_output = run_post_test_checks(
        test_family=test_family,
        fitted_model=fitted_model,
        df=cleaned_df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        test_name=test_name,
    )

    critic_output_dict = critic_output.model_dump()

    # ── Format response via LLM ──
    if not critic_output.checks_applicable:
        final_response = (
            f"Post-test model checks are not applicable for "
            f"{test_name or 'this test'}. Proceeding to final report."
        )
    else:
        format_prompt = f"""Format these post-test model check results clearly for the user.
Use ✅ for passed, ❌ for failed, ⚠️ for warnings.
For each check show: name, method used, key statistic, and plain English verdict.
End with a clear summary of how many passed/failed and what action is recommended.

Results:
{critic_output.summary_message}

Has failures: {critic_output.has_failures}
Failed count: {critic_output.failed_count}
Warning count: {critic_output.warning_count}"""

        response = model.invoke([HumanMessage(content=format_prompt)])
        final_response = response.content.strip()

    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        checks = critic_output_dict.get("checks", [])
        failed_checks = [c for c in checks if not c.get("passed")]
        warned_checks = [c for c in checks if c.get("warning")]

        run.add_metadata({
            # Test context
            "test_name": test_name,
            "test_family": test_family,

            # Applicability — tells you how often critic actually ran vs was skipped
            "checks_applicable": critic_output.checks_applicable,
            "fitted_model_present": fitted_model is not None,

            # Diagnostic results
            "checks_run": len(checks),
            "passed_count": critic_output_dict.get("passed_count", 0),
            "failed_count": critic_output.failed_count,
            "warning_count": critic_output.warning_count,
            "failed_checks": [c.get("check_name") for c in failed_checks],
            "warned_checks": [c.get("check_name") for c in warned_checks],

            # Routing signal — does the graph loop back to rectification?
            "has_failures": critic_output.has_failures,
            "rectification_triggered": critic_output.has_failures and critic_output.checks_applicable,
        })

    return {
        "messages":       [],
        "final_response": final_response,
        "critic_output":  critic_output_dict,
    }