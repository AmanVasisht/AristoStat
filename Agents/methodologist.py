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
from langsmith import traceable, get_current_run_tree

from Prompts.methodologist import METHODOLOGIST_SYSTEM_PROMPT
from Tools.methodologist import (
    METHODOLOGIST_TOOLS,
    init_methodologist_store,
    get_methodologist_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

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

@traceable(
    name="methodologist",
    tags=["agent", "react", "test-selection"],
    metadata={"agent_pattern": "react", "model": "qwen3-32b"}
)
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

    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        run.add_metadata({
            "selected_test": methodologist_output_dict.get("selected_test"),
            "test_family": methodologist_output_dict.get("test_family"),
            "mismatch_warning": methodologist_output_dict.get("mismatch_warning"),
            "bypass_used": intent_output.get("methodologist_bypass", False),
            "react_steps": len(result["messages"]),
        })

    return {
        "messages": result["messages"],
        "final_response": final_response,
        "methodologist_output": methodologist_output_dict,
    }