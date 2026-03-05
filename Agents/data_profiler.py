"""
FILE: Agents/data_profiler.py
------------------------------
Deterministic data profiler — no ReAct loop.
Python handles all computation, LLM is called once for summarisation only.
"""

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from core.profiler_engine import profile_dataframe
from Prompts.data_profiler import DATA_PROFILER_SYSTEM_PROMPT


# ─────────────────────────────────────────────
# LLM — single instance, used only for summary
# ─────────────────────────────────────────────

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ─────────────────────────────────────────────
# PRIVATE — LLM summarisation call
# ─────────────────────────────────────────────

def _llm_summarise(profiler_output: dict, user_message: str = None) -> str:
    """
    Single LLM call — receives the real profiler_output dict and
    generates a human-readable summary. Numbers come from Python,
    not from LLM reasoning.
    """
    user_content = (
        f"Here is the profiler output as JSON:\n\n"
        f"{profiler_output}\n\n"
    )

    response = model.invoke([
        SystemMessage(content=DATA_PROFILER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    return response.content


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_data_profiler(filepath: str, user_message: str = None) -> dict:
    """
    Entry point called by the LangGraph orchestrator.
    
    1. Load CSV with pandas — deterministic
    2. Run profiler engine — deterministic  
    3. Call LLM once for summary — single call, no tool loop
    """
    # Step 1 — Load
    df = pd.read_csv(filepath)

    # Step 2 — Profile (pure Python, no LLM)
    profiler_result = profile_dataframe(df)
    profiler_output_dict = profiler_result.model_dump()

    # Step 3 — Summarise (single LLM call)
    final_response = _llm_summarise(profiler_output_dict, user_message)

    return {
        "final_response": final_response,
        "profiler_output": profiler_output_dict,
    }