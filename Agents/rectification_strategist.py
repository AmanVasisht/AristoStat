"""
FILE: agents/rectification_strategist.py
------------------------------------------
Rectification Strategist — proper ReAct agent.

Architecture:
  - Tools do ALL data fetching (VIF scores, proposals, column resolution, apply)
  - LLM reasons over tool outputs and decides what to recommend
  - LLM never invents data — it only reasons over what tools return
  - ReAct loop: Thought → Tool call → Observation → Thought → ...

Tool call sequence the agent follows:
  1. check_attempt_limit      → know if we're at max attempts
  2. get_failure_context      → which assumptions failed, column roles, n_rows
  3. get_violation_details    → exact VIF scores, statistics — the reasoning fuel
  4. get_proposed_solutions   → fetch options from RECTIFICATION_REGISTRY
  5. (present recommendation + all options to user — this happens in confirm node)
  6. resolve_columns_to_drop  → if user chose drop, parse their free text
  7. apply_chosen_solution    → apply the solution
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langsmith import traceable, get_current_run_tree

from Prompts.rectification_strategist import RECTIFICATION_STRATEGIST_SYSTEM_PROMPT
from Tools.rectification_strategist import (
    RECTIFICATION_TOOLS,
    init_rectification_store,
    get_rectification_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

model = ChatGroq(model="qwen/qwen3-32b", temperature=0)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_rectification_agent():
    return create_react_agent(
        model=model,
        tools=RECTIFICATION_TOOLS,
        prompt=RECTIFICATION_STRATEGIST_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

@traceable(
    name="rectification_strategist",
    tags=["agent", "react", "rectification"],
    metadata={"agent_pattern": "react", "model": "qwen3-32b"}
)
def run_rectification_strategist(
    failed_assumptions: list[str],
    phase: str,
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
    rectification_attempt: int = 1,
    max_attempts: int = 3,
    checker_output: dict | None = None,
    critic_output: dict | None = None,
) -> dict[str, Any]:
    """
    ReAct agent entry point.

    The agent uses tools to fetch real data (VIF scores, proposals, statistics)
    and reasons over that data to produce a recommendation.
    Nothing is hallucinated — all numbers come from tool returns.

    Returns:
        {
          "messages":              Full ReAct message history.
          "final_response":        Agent's recommendation + all options, shown to user.
          "rectification_output":  RectificationOutput dict with proposed_solutions.
          "rectified_df":          DataFrame (unchanged at this stage — apply happens in confirm node).
        }
    """
    # ── Initialise store — tools read from this ──
    init_rectification_store(
        failed_assumptions=failed_assumptions,
        phase=phase,
        cleaned_df=cleaned_df,
        methodologist_output=methodologist_output,
        rectification_attempt=rectification_attempt,
        max_attempts=max_attempts,
        checker_output=checker_output or {},
        critic_output=critic_output or {},
    )

    agent = create_rectification_agent()

    content = (
        f"Assumption failures detected during "
        f"{'pre-test' if phase == 'pre_test' else 'post-test'} checking.\n\n"
        f"Your task:\n"
        f"1. Call check_attempt_limit to check if we are at max attempts.\n"
        f"2. Call get_failure_context to understand which assumptions failed.\n"
        f"3. Call get_violation_details to get exact statistics (VIF scores, p-values, etc.).\n"
        f"4. Call get_proposed_solutions to get the available options from the registry.\n"
        f"5. Reason over the real data from the tools — do NOT invent numbers.\n"
        f"6. Present your recommendation with 'I recommend Option N because [specific reason "
        f"tied to actual values from tools]', followed by all options.\n\n"
        f"Do NOT call apply_chosen_solution or resolve_columns_to_drop yet — "
        f"wait for the user to confirm their choice."
    )

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final response from last AIMessage ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            if msg.content and not getattr(msg, "tool_calls", None):
                final_response = msg.content
                break

    # ── Pull rectification_output from store ──
    store = get_rectification_store()
    rect_output = store.get("rectification_output")
    rect_output_dict = rect_output.model_dump() if rect_output else {}

    if not rect_output_dict.get("proposed_solutions"):
        from Schemas.rectification_strategist import RectificationPhase as RP
        from core.rectification_engine import build_rectification_output
        try:
            ph = RP(phase)
        except ValueError:
            ph = RP.PRE_TEST
        _, proposals_output = build_rectification_output(
            failed_assumptions=failed_assumptions,
            phase=ph,
            chosen_solution_id=None,
            df=cleaned_df,
            dependent_var=methodologist_output.get("dependent_variable"),
            independent_vars=methodologist_output.get("independent_variables", []).copy(),
            rectification_attempt=rectification_attempt,
            max_attempts=max_attempts,
        )
        rect_output_dict = proposals_output.model_dump()

    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        proposed_solutions = rect_output_dict.get("proposed_solutions", [])

        run.add_metadata({
            # Phase context — pre_test vs post_test tells you which loop triggered this
            "phase": phase,
            "rectification_attempt": rectification_attempt,
            "max_attempts": max_attempts,
            "at_attempt_limit": rectification_attempt >= max_attempts,

            # What failed — the reason this agent was invoked
            "failed_assumptions": failed_assumptions,
            "failed_count": len(failed_assumptions),
            "original_test": methodologist_output.get("selected_test"),

            # What solutions were proposed
            "proposed_solutions_count": len(proposed_solutions),
            "proposed_solution_ids": [
                s.get("solution_id") for s in proposed_solutions
            ],

            # Solution applied (if agent called apply_chosen_solution)
            "solution_applied": rect_output_dict.get("chosen_solution_id"),
            "test_switched": rect_output_dict.get("test_switched", False),
            "new_test": rect_output_dict.get("new_test"),
            "columns_dropped": rect_output_dict.get("columns_dropped", []),
            "transformation_applied": rect_output_dict.get("transformation_applied"),

            # ReAct loop depth — useful for spotting runaway loops
            "react_steps": len(result["messages"]),
        })

    return {
        "messages":             result["messages"],
        "final_response":       final_response,
        "rectification_output": rect_output_dict,
        "rectified_df":         cleaned_df,
    }