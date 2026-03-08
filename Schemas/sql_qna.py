"""
FILE: Schemas/sql_qna.py
-------------------------
Pydantic output schema for the SQL QnA Agent.

Design notes:
    - filtered_df is intentionally included for future planner
      integration (Use Case 2: filter then analyse).
      Currently always None — will be populated when planner
      routes filter-then-analyse queries to the stats pipeline.
"""

from pydantic import BaseModel, Field


class SqlQnAOutput(BaseModel):
    """
    Structured output from the SQL QnA agent.
    """

    query_understood: str = Field(
        description="What the agent understood the user was asking"
    )
    generated_sql: str = Field(
        description="The DuckDB SQL query that was executed"
    )
    result_table: list[dict] = Field(
        description="Raw query results as list of row dicts"
    )
    row_count: int = Field(
        description="Number of rows returned by the query"
    )
    explanation: str = Field(
        description="Plain English answer to the user's question"
    )
    fatal_error: str | None = Field(
        default=None,
        description="Error message if agent failed to answer"
    )

    # ── Future planner integration — Use Case 2 (filter then analyse) ──
    # Populated when SQL result is a filtered dataset intended
    # for the statistics pipeline. Ignored in current pure-QnA flow.
    filtered_df: str | None = Field(
        default=None,
        description="Serialised filtered DataFrame for stats pipeline (future use)"
    )