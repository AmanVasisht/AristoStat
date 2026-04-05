"""
FILE: agents/final_report.py
------------------------------
Final Report agent — wires together all components and exposes
the public run_final_report() entry point for the LangGraph orchestrator.

This is the last agent in the pipeline. It receives all upstream outputs,
assembles a structured report, renders it in chat as markdown, and
generates a downloadable .docx Word document.

Imports:
  - LLM model
  - Prompt   ← prompts/final_report_prompt.py
  - Tools    ← tools/final_report_tools.py
  - Engine   ← core/final_report_engine.py  (called via tools)
  - Schemas  ← schemas/final_report_schema.py
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langsmith import traceable, get_current_run_tree

from Prompts.final_report import FINAL_REPORT_SYSTEM_PROMPT
from Tools.final_report import (
    FINAL_REPORT_TOOLS,
    init_report_store,
    get_report_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_final_report_agent():
    """Create and return the Final Report ReAct agent."""
    return create_react_agent(
        model=model,
        tools=FINAL_REPORT_TOOLS,
        prompt=FINAL_REPORT_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

@traceable(
    name="final_report",
    tags=["agent", "react", "report-generation"],
    metadata={"agent_pattern": "react", "model": "llama-4-maverick-17b"}
)
def run_final_report(
    original_query: str,
    profiler_output: dict,
    preprocessor_output: dict,
    methodologist_output: dict,
    checker_output: dict,
    statistician_output: dict,
    rectification_output: dict | None = None,
    critic_output: dict | None = None,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    This is the terminal node — nothing follows it.

    Args:
        original_query:       The user's original question/request.
        profiler_output:      ProfilerOutput dict.
        preprocessor_output:  PreprocessorOutput dict.
        methodologist_output: MethodologistOutput dict.
        checker_output:       AssumptionCheckerOutput dict.
        statistician_output:  StatisticianOutput dict.
        rectification_output: RectificationOutput dict (None if no failures).
        critic_output:        ModelCriticOutput dict (None if non-regression test).

    Returns:
        {
          "messages":       Full LangGraph message history.
          "final_response": The complete markdown report shown in chat.
          "report_output":  Raw FinalReportOutput dict.
                            Key fields:
                              - markdown_report: full markdown string
                              - docx_path:       path to generated .docx file
                              - docx_generated:  bool
                              - caveats:         list of caveats
        }
    """
    init_report_store(
        original_query=original_query,
        profiler_output=profiler_output,
        preprocessor_output=preprocessor_output,
        methodologist_output=methodologist_output,
        checker_output=checker_output,
        statistician_output=statistician_output,
        rectification_output=rectification_output,
        critic_output=critic_output,
    )

    agent = create_final_report_agent()
    content = "Please generate the final analysis report."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve report output from store ──
    store = get_report_store()
    report_output = store.get("report_output")
    report_output_dict = report_output.model_dump() if report_output else {}

    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        # Summarise what the full pipeline went through — visible on this terminal trace
        p_value = statistician_output.get("p_value")

        run.add_metadata({
            # Report generation health
            "docx_generated": report_output_dict.get("docx_generated", False),
            "docx_path": report_output_dict.get("docx_path"),
            "caveats_count": len(report_output_dict.get("caveats", [])),
            "react_steps": len(result["messages"]),

            # Full pipeline summary — useful for end-to-end filtering
            "original_query": original_query,
            "test_executed": statistician_output.get("test_name"),
            "p_value": p_value,
            "significant": p_value < 0.05 if p_value is not None else None,
            "rectification_occurred": rectification_output is not None,
            "critic_occurred": critic_output is not None,
            "assumption_failures": checker_output.get("failed_count", 0),
        })

    return {
        "messages":       result["messages"],
        "final_response": final_response,
        "report_output":  report_output_dict,
    }