"""
FILE: agents/methodologist.py
-------------------------------
Methodologist agent — wires together all components and exposes
the public run_methodologist() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/methodologist_prompt.py
  - Tools   ← tools/methodologist_tools.py
  - Engine  ← core/methodologist_engine.py  (called via tools)
  - Schemas ← schemas/methodologist_schema.py (MethodologistOutput in LangGraph state)
"""

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from Prompts.methodologist import METHODOLOGIST_SYSTEM_PROMPT
from Tools.methodologist import (
    METHODOLOGIST_TOOLS,
    init_methodologist_store,
    get_methodologist_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

from langchain_groq import ChatGroq
model = ChatGroq(model="qwen/qwen3-32b", temperature=0)

# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_methodologist_agent():
    """
    Create and return the Methodologist ReAct agent.
    Called fresh per invocation to avoid stale state.
    """
    return create_react_agent(
        model=model,
        tools=METHODOLOGIST_TOOLS,
        prompt=METHODOLOGIST_SYSTEM_PROMPT
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_methodologist(
    intent_output: dict,
    profiler_output: dict,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        intent_output:   The IntentOutput dict from the Intent Interpreter agent.
        profiler_output: The ProfilerOutput dict from the Data Profiler agent.

    Returns:
        {
          "messages":             Full LangGraph message history.
          "final_response":       Human-readable test selection shown to user.
          "methodologist_output": Raw MethodologistOutput dict for downstream agents.
                                  Contains selected_test, column roles, reasoning,
                                  and mismatch_warning if applicable.
        }
    """
    # Seed the store before invoking the agent
    init_methodologist_store(
        intent_output=intent_output,
        profiler_output=profiler_output,
    )

    agent = create_methodologist_agent()

    content = "Please select the appropriate statistical test for this analysis."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve MethodologistOutput from store ──
    store = get_methodologist_store()
    methodologist_output = store.get("methodologist_output")
    methodologist_output_dict = (
        methodologist_output.model_dump() if methodologist_output else {}
    )

    return {
        "messages": result["messages"],
        "final_response": final_response,
        "methodologist_output": methodologist_output_dict,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/methodologist.py
# ─────────────────────────────────────────────

# if __name__ == "__main__":

#     # ── Mock IntentOutput for standalone testing ──
#     mock_intent_output = {
#         "intent_type":          "column_relationship",
#         "analysis_goal":        "inference",
#         "confidence":           "high",
#         "methodologist_bypass": False,
#         "requested_test":       None,
#         "columns": [
#             {"name": "salary",  "role": "dependent",  "dtype_from_profiler": "continuous"},
#             {"name": "gender",  "role": "grouping",   "dtype_from_profiler": "categorical"},
#         ],
#         "all_columns_mode":          False,
#         "suggested_combinations":    [],
#         "invalid_columns":           [],
#         "column_warnings":           [],
#         "interpretation_summary":    "Check if salary differs significantly between genders.",
#         "clarification_needed":      False,
#         "clarification_question":    None,
#         "original_query":            "Does salary differ between genders?",
#     }

#     # ── Mock ProfilerOutput for standalone testing ──
#     mock_profiler_output = {
#         "n_rows": 500,
#         "n_cols": 4,
#         "continuous_columns": [
#             {"column": "salary", "mean": 55000, "std": 12000, "skewness": 1.2,
#              "missing_pct": 0.01, "missing_severity": "low", "anomaly_count": 5},
#         ],
#         "categorical_columns": [
#             {"column": "gender", "cardinality": 2, "mode": "M",
#              "mode_frequency": 0.54, "missing_pct": 0.0,
#              "missing_severity": "low", "class_imbalance_flag": False},
#         ],
#         "warnings": [],
#         "fatal_errors": [],
#     }

#     print("\n" + "=" * 60)
#     print("  ARISTOSTAT — Methodologist Agent")
#     print("=" * 60 + "\n")

#     output = run_methodologist(
#         intent_output=mock_intent_output,
#         profiler_output=mock_profiler_output,
#     )

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     print("Selected test:  ", output["methodologist_output"].get("selected_test"))
#     print("Selection mode: ", output["methodologist_output"].get("selection_mode"))
#     print("Mismatch warning:", output["methodologist_output"].get("mismatch_warning"))