"""
FILE: Tools/sql_qna.py
-----------------------
Tools and module-level store for the SQL QnA Agent.

Tools:
    get_schema()     — inspect CSV columns, types, sample values
    execute_sql()    — run DuckDB SQL against the CSV
    fix_and_retry()  — self-correct SQL once on failure

Store:
    _sql_store       — runtime context shared across tools
                       during a single agent run. Same pattern
                       as Tools/rectification_strategist.py.

Design notes:
    - DuckDB queries the CSV directly as table 'data'.
      No database setup required — works for any uploaded CSV.
    - fix_and_retry() has a retry_used guard — agent can call
      it at most once per run to prevent infinite correction loops.
    - get_sql_store() is exposed so the agent can read final
      results after the ReAct loop completes.
"""

import duckdb
import pandas as pd
from typing import Any

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


# ─────────────────────────────────────────────
# MODULE-LEVEL STORE
# ─────────────────────────────────────────────

_sql_store: dict[str, Any] = {}


def init_sql_store(csv_path: str, user_question: str) -> None:
    """
    Initialise the store before each agent run.
    Called from Agents/sql_qna.py before the ReAct loop starts.
    """
    _sql_store.clear()
    _sql_store["csv_path"]      = csv_path
    _sql_store["user_question"] = user_question
    _sql_store["generated_sql"] = ""
    _sql_store["result_table"]  = []
    _sql_store["row_count"]     = 0
    _sql_store["retry_used"]    = False
    _sql_store["fatal_error"]   = None


def get_sql_store() -> dict[str, Any]:
    """
    Returns the current store state.
    Called from Agents/sql_qna.py after the ReAct loop
    to build the final SqlQnAOutput.
    """
    return _sql_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_schema() -> str:
    """
    Inspect the CSV file to retrieve column names, data types,
    and a small sample of values for each column.
    Always call this FIRST before writing any SQL query.
    Returns table name, total row count, and per-column details.
    """
    csv_path = _sql_store.get("csv_path")
    if not csv_path:
        return "ERROR: No CSV path found in store."

    try:
        df         = pd.read_csv(csv_path, nrows=5)
        full_df    = pd.read_csv(csv_path)
        total_rows = len(full_df)

        schema_lines = []
        for col in df.columns:
            dtype       = str(df[col].dtype)
            sample_vals = df[col].dropna().tolist()[:3]
            schema_lines.append(
                f"  - {col} ({dtype}): sample values = {sample_vals}"
            )

        return (
            f"Table name to use in all SQL queries: 'data'\n"
            f"Total rows in dataset: {total_rows}\n"
            f"Columns:\n" + "\n".join(schema_lines)
        )

    except Exception as e:
        return f"ERROR reading schema: {str(e)}"


@tool
def execute_sql(query: str) -> str:
    """
    Execute a DuckDB SQL query against the uploaded CSV file.
    The CSV is registered as table 'data' — always use 'data'
    as the table name, never the actual filename.

    Args:
        query: A valid DuckDB SQL query using 'data' as table name.

    Returns:
        Query results as a readable string preview, or an error message.
    """
    csv_path = _sql_store.get("csv_path")
    if not csv_path:
        return "ERROR: No CSV path found in store."

    # Save the query so it appears in the output schema
    _sql_store["generated_sql"] = query

    try:
        conn = duckdb.connect()

        # Register CSV as table 'data' — works for any filename
        conn.execute(
            f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{csv_path}')"
        )

        result_df = conn.execute(query).df()
        conn.close()

        if result_df.empty:
            _sql_store["result_table"] = []
            _sql_store["row_count"]    = 0
            return (
                "Query executed successfully but returned 0 rows. "
                "The dataset may not contain records matching your conditions."
            )

        # Store full results for output schema
        result_records             = result_df.to_dict(orient="records")
        _sql_store["result_table"] = result_records
        _sql_store["row_count"]    = len(result_records)

        # Cap preview at 50 rows to avoid overflowing LLM context
        preview = result_df.head(50).to_string(index=False)

        return (
            f"Query executed successfully.\n"
            f"Total rows returned: {len(result_records)}\n"
            f"Results preview (up to 50 rows):\n{preview}"
        )

    except Exception as e:
        error_msg = str(e)
        _sql_store["fatal_error"] = error_msg
        return f"SQL ERROR: {error_msg}"


@tool
def fix_and_retry(original_query: str, error_message: str) -> str:
    """
    Call this when execute_sql() returns an error.
    Uses an LLM to self-correct the SQL and re-executes it once.
    Only call this ONCE per run — further retries are blocked.

    Args:
        original_query: The SQL query that failed.
        error_message:  The exact error returned by execute_sql().

    Returns:
        Result of the corrected query execution, or an error message.
    """
    if _sql_store.get("retry_used"):
        return (
            "ERROR: Retry already used once this run. "
            "Cannot attempt further corrections."
        )

    _sql_store["retry_used"] = True

    csv_path = _sql_store.get("csv_path")
    if not csv_path:
        return "ERROR: No CSV path found in store."

    # Get fresh schema for correction context
    schema_info = get_schema.invoke({})

    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

    fix_prompt = f"""You wrote this DuckDB SQL query:
{original_query}

It failed with this error:
{error_message}

The table schema is:
{schema_info}

Fix the query and return ONLY the corrected SQL.
Rules:
- Table name must be 'data'
- Use valid DuckDB SQL syntax
- No explanation, no markdown, no backticks — raw SQL only"""

    response    = llm.invoke([HumanMessage(content=fix_prompt)])
    fixed_query = response.content.strip().strip("```sql").strip("```").strip()

    print(f"[fix_and_retry] corrected query: {fixed_query}")

    # Re-execute with the corrected query
    return execute_sql.invoke({"query": fixed_query})