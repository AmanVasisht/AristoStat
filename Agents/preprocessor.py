"""
FILE: agents/preprocessor.py
------------------------------
Preprocessor agent — wires together all components and exposes
the public run_preprocessor() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/preprocessor_prompt.py
  - Tools   ← tools/preprocessor_tools.py
  - Engine  ← core/preprocessor_engine.py  (called via tools)
  - Schemas ← schemas/preprocessor_schema.py

Note on state passing:
  The cleaned DataFrame is stored in two ways:
    1. As a pd.DataFrame in memory (cleaned_df) — used by the next agent directly
       if running in the same process (recommended for performance).
    2. As a JSON string in PreprocessorOutput.cleaned_data_json — travels in
       LangGraph state and can be reconstructed with pd.read_json() anywhere.
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langsmith import traceable, get_current_run_tree

from core.preprocessor_engine import preprocess_dataframe


# ─────────────────────────────────────────────
# LLM — used only for formatting the summary
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

@traceable(
    name="preprocessor",
    tags=["agent", "direct-call", "preprocessing"],
    metadata={"agent_pattern": "direct", "model": "llama-3.1-8b-instant"}
)
def run_preprocessor(
    raw_df: pd.DataFrame,
    profiler_output: dict,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    Calls the preprocessor engine directly — no LLM needed for cleaning.
    The LLM is only used to format the final human-readable summary.
    """
    # ── Capture shape before cleaning ──
    rows_before = len(raw_df)
    cols_before = len(raw_df.columns)

    cleaned_df, preprocessor_output = preprocess_dataframe(
        df=raw_df,
        profiler_output=profiler_output,
    )

    preprocessor_output_dict = preprocessor_output.model_dump()

    # ── Format human-readable summary via LLM ──
    changes = preprocessor_output.changes_summary
    warnings = preprocessor_output.warnings
    fatal = preprocessor_output.fatal_error

    if fatal:
        final_response = f"⚠️ Fatal error during preprocessing: {fatal}"
    elif not changes:
        final_response = "Your data required no cleaning — it is already in good shape."
    else:
        summary_prompt = f"""Summarize these data cleaning steps in 3-5 plain English sentences for a non-technical user.
Be concise. List what was done and the final dataset shape.

Changes made: {changes}
Warnings: {warnings}
Original shape: {preprocessor_output.original_shape}
Final shape: {preprocessor_output.final_shape}
Rows dropped: {preprocessor_output.rows_dropped_total}"""

        response = model.invoke([HumanMessage(content=summary_prompt)])
        final_response = response.content.strip()

    # ── LangSmith metadata ──
    run = get_current_run_tree()
    if run:
        run.add_metadata({
            "rows_before": rows_before,
            "cols_before": cols_before,
            "rows_after": len(cleaned_df),
            "cols_after": len(cleaned_df.columns),
            "rows_dropped": preprocessor_output.rows_dropped_total,
            "changes_made": len(changes) if changes else 0,
            "warnings_count": len(warnings) if warnings else 0,
            "fatal_error": bool(fatal),
            "cleaning_required": bool(changes),
        })

    return {
        "messages":            [],
        "final_response":      final_response,
        "preprocessor_output": preprocessor_output_dict,
        "cleaned_df":          cleaned_df,
    }