"""
FILE: agents/intent_interpreter.py
-------------------------------------
Intent Interpreter agent — wires together all components and exposes
the public run_intent_interpreter() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/intent_interpreter_prompt.py
  - Tools   ← tools/intent_interpreter_tools.py
  - Engine  ← core/intent_engine.py  (used via tools)
  - Schemas ← schemas/intent_schema.py (IntentOutput travels in LangGraph state)
"""

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langsmith import traceable, get_current_run_tree


from Prompts.intent_interpreter import INTENT_INTERPRETER_SYSTEM_PROMPT
from Tools.intent_interpreter import (
    INTENT_INTERPRETER_TOOLS,
    init_intent_store,
    get_intent_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

from langchain_groq import ChatGroq
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
)
# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_intent_interpreter_agent():
    """
    Create and return the Intent Interpreter ReAct agent.
    Called fresh per invocation to avoid stale state.
    """
    return create_react_agent(
        model=model,
        tools=INTENT_INTERPRETER_TOOLS,
        prompt=INTENT_INTERPRETER_SYSTEM_PROMPT
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────


@traceable(
    name="intent_interpreter",
    tags=["agent", "react", "intent-parsing"],
    metadata={"agent_pattern": "react", "model": "qwen3-32b"}
)
def run_intent_interpreter(
    profiler_output: dict,
    user_query: str,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        profiler_output: The ProfilerOutput dict from the Data Profiler agent.
        user_query:      The raw query string from the user.

    Returns:
        {
          "messages":      Full LangGraph message history.
          "final_response": Human-readable interpretation summary shown to user.
          "intent_output":  Raw IntentOutput dict for downstream agents.
                            Contains methodologist_bypass flag, columns, goal, etc.
        }
    """
    # Seed the store before invoking the agent
    init_intent_store(
        profiler_output=profiler_output,
        original_query=user_query,
    )

    agent = create_intent_interpreter_agent()

    # The agent retrieves query and profiler output via tools,
    # so we just need a simple trigger message here.
    content = "Please interpret the user's analysis intent and produce a structured output."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve validated IntentOutput from store ──
    store = get_intent_store()
    intent_output = store.get("intent_output")
    intent_output_dict = intent_output.model_dump() if intent_output else {}
    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        run.add_metadata({
            "user_query": user_query,
            "analysis_type": intent_output_dict.get("analysis_type"),
            "target_variable": intent_output_dict.get("target_variable"),
            "predictor_variables": intent_output_dict.get("predictor_variables"),
            "methodologist_bypass": intent_output_dict.get("methodologist_bypass", False),
            "react_steps": len(result["messages"]),
        })
    return {
        "messages": result["messages"],
        "final_response": final_response,
        "intent_output": intent_output_dict,
    }

