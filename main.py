"""
FILE: main.py
--------------
LangGraph orchestrator for Aristostat.
Wires all nine Agents into a StateGraph with conditional edges
and human-in-the-loop interrupt() checkpoints.

Pipeline flow:
  data_profiler
      ↓ [interrupt — show stats, confirm proceed]
  intent_interpreter_run → intent_interpreter_confirm
      ↓ [interrupt — confirm interpretation]
  methodologist_run → methodologist_confirm
      ↓ [interrupt — confirm test selection]
  preprocessor_run → preprocessor_confirm
      ↓ [interrupt — show cleaning summary]
  assumption_checker_run → assumption_checker_confirm
      ↓ [interrupt — show pass/fail, rectify or proceed?]
      ↓ if failures → rectification_strategist_run → rectification_strategist_confirm
          ↓ if test_switch → back to assumption_checker_run (loop, max 3)
          ↓ if proceed/correction → statistician
      ↓ if all pass → statistician
  statistician
      ↓
  model_critic_run → model_critic_confirm
      ↓ [interrupt if failures — rectify or proceed?]
      ↓ if failures → rectification_strategist_run → rectification_strategist_confirm
      ↓ if pass/skipped → final_report
  final_report
      ↓ [END]
"""

import json
import pandas as pd
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import time
import io

# ── Agent entry points ──
from Agents.data_profiler            import run_data_profiler
from Agents.intent_interpreter       import run_intent_interpreter
from Agents.methodologist            import run_methodologist
from Agents.preprocessor             import run_preprocessor
from Agents.assumption_checker       import run_assumption_checker
from Agents.rectification_strategist import run_rectification_strategist
from Agents.statistician             import run_statistician
from Agents.critic                   import run_model_critic
from Agents.final_report             import run_final_report

# ── Fitted model accessor ──
from Tools.statistician import get_fitted_model
from Schemas.methodologist import MethodologistOutput, SelectionMode


# ─────────────────────────────────────────────
# SERIALIZATION HELPERS
# ─────────────────────────────────────────────

def _serialize_output(obj) -> dict:
    """Converts any Pydantic model or enum-containing dict to plain JSON-safe dict."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump_json"):
        return json.loads(obj.model_dump_json())
    return json.loads(json.dumps(obj, default=lambda o: o.value if hasattr(o, "value") else str(o)))


def _df_to_state(df) -> str | None:
    """Serialize a DataFrame to JSON string for state storage."""
    if df is None:
        return None
    return df.to_json(orient="split")


def _df_from_state(s: str | None):
    """Deserialize a DataFrame from JSON string in state."""
    if s is None:
        return None
    return pd.read_json(io.StringIO(s), orient="split")


# ─────────────────────────────────────────────
# STATE SCHEMA
# ─────────────────────────────────────────────

class AristostatState(TypedDict, total=False):
    # ── Inputs ──
    csv_path:             str
    user_query:           str

    # ── Working data (JSON strings for MemorySaver serialization) ──
    raw_df:               str
    cleaned_df:           str
    rectified_df:         str

    # ── Agent outputs ──
    profiler_output:      dict
    intent_output:        dict
    methodologist_output: dict
    preprocessor_output:  dict
    checker_output:       dict
    rectification_output: dict | None
    statistician_output:  dict
    critic_output:        dict | None
    report_output:        dict

    # ── Run/confirm handoff fields ──
    _intent_response:          str
    _methodologist_response:   str
    _preprocessor_response:    str
    _checker_response:         str
    _rectification_response:   str
    _rectification_failed:     list
    _rectification_df_json:    str
    _critic_response:          str

    # ── Rectification loop ──
    rectification_attempt:     int
    rectification_phase:       str

    # ── Human checkpoint responses ──
    user_confirmed_proceed:    bool
    user_rectify_or_proceed:   str
    user_chosen_solution_id:   str | None

    # ── Routing flags ──
    fatal_error:               str | None
    bypass_assumption_checker: bool


# ─────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────

def node_data_profiler(state: AristostatState) -> AristostatState:
    """Loads CSV, profiles dataset, shows stats to user."""
    print("\n[NODE: data_profiler] Starting...")
    result = run_data_profiler(
        filepath=state["csv_path"],
        user_message=state.get("user_query"),
    )
    print(f"[NODE: data_profiler] profiler_output keys: {list(result.get('profiler_output', {}).keys())}")
    print(f"[NODE: data_profiler] fatal_errors: {result.get('profiler_output', {}).get('fatal_errors')}")
    print(f"[NODE: data_profiler] warnings count: {len(result.get('profiler_output', {}).get('warnings', []))}")

    try:
        raw_df = pd.read_csv(state["csv_path"])
        print(f"[NODE: data_profiler] raw_df shape: {raw_df.shape}")
    except Exception as e:
        print(f"[NODE: data_profiler] ERROR loading csv: {e}")
        raw_df = None

    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Shall I proceed with the analysis? (yes/no)",
        "type":    "confirm",
    })

    if str(user_response).strip().lower() not in ("yes", "y"):
        print("[NODE: data_profiler] User stopped pipeline.")
        return {**state, "fatal_error": "User stopped pipeline after data profiling."}

    profiler_serialized = _serialize_output(result["profiler_output"])
    print(f"[NODE: data_profiler] profiler_serialized keys: {list(profiler_serialized.keys())}")

    return {
        **state,
        "raw_df":          _df_to_state(raw_df),
        "profiler_output": profiler_serialized,
    }


# ── INTENT INTERPRETER ──

def node_intent_interpreter_run(state: AristostatState) -> AristostatState:
    """Runs intent interpreter LLM — no interrupts."""
    print("\n[NODE: intent_interpreter_run] Starting...")
    print(f"[NODE: intent_interpreter_run] user_query: {state.get('user_query')}")
    print(f"[NODE: intent_interpreter_run] profiler_output keys: {list(state.get('profiler_output', {}).keys())}")
    

    result = run_intent_interpreter(
        user_query=state["user_query"],
        profiler_output=state["profiler_output"],
    )

    print(f"[NODE: intent_interpreter_run] result keys: {list(result.keys())}")
    intent = result.get("intent_output", {})
    print(f"[NODE: intent_interpreter_run] raw intent_output: {intent}")

    if intent.get("fatal_error"):
        print(f"[NODE: intent_interpreter_run] FATAL: {intent.get('fatal_error')}")
        return {**state, "fatal_error": result["final_response"]}

    intent_serialized = _serialize_output(intent)
    print(f"[NODE: intent_interpreter_run] intent_serialized: {intent_serialized}")

    return {
        **state,
        "intent_output":     intent_serialized,
        "_intent_response":  result["final_response"],
    }


def node_intent_interpreter_confirm(state: AristostatState) -> AristostatState:
    """Shows intent result to user and confirms — contains the interrupt."""
    print("\n[NODE: intent_interpreter_confirm] Starting...")
    print(f"[NODE: intent_interpreter_confirm] intent_output: {state.get('intent_output')}")

    if state.get("fatal_error"):
        interrupt({
            "message": state["fatal_error"],
            "prompt":  "Please correct the column name and re-submit your query.",
            "type":    "fatal_error",
        })
        return state

    # ── Open-ended: user picks a suggested combination ──
    intent = state.get("intent_output", {})
    intent_type = intent.get("intent_type")
    if intent_type == "open_ended" and intent.get("suggested_combinations"):
        combo_response = interrupt({
            "message": state.get("_intent_response", ""),
            "prompt":  "Please choose one of the suggested combinations (enter the number).",
            "type":    "choose_combination",
            "options": intent["suggested_combinations"],
        })
        chosen_query = _resolve_combination_choice(
            combo_response, intent["suggested_combinations"]
        )
        result = run_intent_interpreter(
            user_query=chosen_query,
            profiler_output=state["profiler_output"],
        )
        intent_serialized = _serialize_output(result.get("intent_output", {}))
        return {
            **state,
            "intent_output":    intent_serialized,
            "_intent_response": result["final_response"],
        }

    user_response = interrupt({
        "message": state.get("_intent_response", ""),
        "prompt":  "Does this correctly capture your intent? (yes/no, or describe correction)",
        "type":    "confirm",
    })

    if str(user_response).strip().lower() in ("yes", "y"):
        print("[NODE: intent_interpreter_confirm] User confirmed.")
        return state

    # User corrected — re-run
    print(f"[NODE: intent_interpreter_confirm] User corrected: {user_response}")
    corrected_result = run_intent_interpreter(
        user_query=str(user_response),
        profiler_output=state["profiler_output"],
    )
    corrected_intent = _serialize_output(corrected_result.get("intent_output", {}))
    print(f"[NODE: intent_interpreter_confirm] corrected: {corrected_intent}")
    return {**state, "intent_output": corrected_intent}


# ── METHODOLOGIST ──

def node_methodologist_run(state: AristostatState) -> AristostatState:
    """Runs methodologist LLM — no interrupts."""
    print("\n[NODE: methodologist_run] Starting...")
    print(f"[NODE: methodologist_run] intent_output: {state.get('intent_output')}")
    intent = state.get("intent_output", {})
    if intent.get("methodologist_bypass") and intent.get("requested_test"):

        selected_test = intent["requested_test"]

        dependent = next(
            (c["name"] for c in intent["columns"] if c["role"] == "dependent"),
            None
        )

        independents = [
            c["name"]
            for c in intent["columns"]
            if c["role"] == "independent"
        ]

        # Build REAL schema object (not manual dict)
        methodologist_output_obj = MethodologistOutput(
            selected_test=selected_test,

            # IMPORTANT: must use enum
            selection_mode=SelectionMode.BYPASS,

            dependent_variable=dependent,
            independent_variables=independents,

            reasoning=f"Using {selected_test} based on user's explicit request.",

            original_query=intent.get("original_query", "")
        )

        structured_output = methodologist_output_obj.model_dump()

        return {
            **state,
            "methodologist_output": structured_output,
            "_methodologist_response":
                f"Using {selected_test} based on your explicit request."
        }
    result = run_methodologist(
        intent_output=state["intent_output"],
        profiler_output=state["profiler_output"],
    )

    print(f"[NODE: methodologist_run] result keys: {list(result.keys())}")
    print(f"[NODE: methodologist_run] methodologist_output: {result.get('methodologist_output')}")

    methodologist_serialized = _serialize_output(result["methodologist_output"])
    print(f"[NODE: methodologist_run] methodologist_serialized: {methodologist_serialized}")

    return {
        **state,
        "methodologist_output":    methodologist_serialized,
        "_methodologist_response": result["final_response"],
    }


# def node_methodologist_confirm(state: AristostatState) -> AristostatState:
#     """Shows test selection to user and confirms — contains the interrupt."""
#     print("\n[NODE: methodologist_confirm] Starting...")

#     user_response = interrupt({
#         "message": state.get("_methodologist_response", ""),
#         "prompt":  "Shall I proceed with this test? (yes/no, or describe the test you want)",
#         "type":    "confirm",
#     })

#     user_input = str(user_response).strip().lower()

#     # ── User confirmed ──
#     if user_input in ("yes", "y"):
#         return state

#     # ── User said no with no correction — stop ──
#     if user_input in ("no", "n"):
#         print("[NODE: methodologist_confirm] User stopped pipeline.")
#         return {**state, "fatal_error": "User stopped pipeline after test selection."}

#     # ── User gave a correction — re-run methodologist with their input ──
#     print(f"[NODE: methodologist_confirm] User corrected: {user_response}")
#     from Agents.methodologist import run_methodologist

#     # Build a corrected intent that includes the user's requested test
#     corrected_intent = {
#         **state.get("intent_output", {}),
#         "requested_test":      str(user_response),
#         "methodologist_bypass": True,
#         "intent_type":         "explicit_test",
#     }

#     corrected_result = run_methodologist(
#         intent_output=corrected_intent,
#         profiler_output=state["profiler_output"],
#     )

#     corrected_methodologist = _serialize_output(corrected_result["methodologist_output"])
#     print(f"[NODE: methodologist_confirm] corrected test: {corrected_methodologist.get('selected_test')}")

#     return {
#         **state,
#         "methodologist_output":    corrected_methodologist,
#         "_methodologist_response": corrected_result["final_response"],
#     }


def node_methodologist_confirm(state: AristostatState) -> AristostatState:
    print("\n[NODE: methodologist_confirm] Starting...")

    user_response = interrupt({
        "message": state.get("_methodologist_response", ""),
        "prompt":  "Shall I proceed with this test? (yes/no, or describe the test you want)",
        "type":    "confirm",
    })

    user_input = str(user_response).strip().lower()

    # ── User confirmed ──
    if user_input in ("yes", "y"):
        return state

    # ── User said no with no correction — stop ──
    if user_input in ("no", "n"):
        print("[NODE: methodologist_confirm] User stopped pipeline.")
        return {**state, "fatal_error": "User stopped pipeline after test selection."}

    # ── User gave a correction — extract canonical test name first ──
    print(f"[NODE: methodologist_confirm] User corrected: {user_response}")

    from core.intent_engine import detect_explicit_test
    # from Agents.methodologist import run_methodologist

    # Try to extract a known test name from the user's free text
    # e.g. "no, i said proceed with mlr" → "Multiple Linear Regression"
    canonical_test = detect_explicit_test(str(user_response))

    if canonical_test:
        print(f"[NODE: methodologist_confirm] Extracted test: {canonical_test}")
        requested_test = canonical_test
    else:
        # No known test found — pass the raw string and let the LLM interpret it
        print(f"[NODE: methodologist_confirm] No canonical test found, passing raw: {user_response}")
        requested_test = str(user_response)

    corrected_intent = {
        **state.get("intent_output", {}),
        "requested_test":       requested_test,
        "methodologist_bypass": True,
        "intent_type":          "explicit_test",
    }

    corrected_result = run_methodologist(
        intent_output=corrected_intent,
        profiler_output=state["profiler_output"],
    )

    corrected_methodologist = _serialize_output(corrected_result["methodologist_output"])
    print(f"[NODE: methodologist_confirm] corrected test: {corrected_methodologist.get('selected_test')}")

    return {
        **state,
        "methodologist_output":    corrected_methodologist,
        "_methodologist_response": corrected_result["final_response"],
    }

# ── PREPROCESSOR ──

def node_preprocessor_run(state: AristostatState) -> AristostatState:
    """Runs preprocessor LLM — no interrupts."""
    print("\n[NODE: preprocessor_run] Starting...")
    

    raw_df = _df_from_state(state["raw_df"])
    print(f"[NODE: preprocessor_run] raw_df shape: {raw_df.shape if raw_df is not None else 'None'}")

    result = run_preprocessor(
        raw_df=raw_df,
        profiler_output=state["profiler_output"],
    )

    print(f"[NODE: preprocessor_run] result keys: {list(result.keys())}")
    print(f"[NODE: preprocessor_run] fatal_error: {result['preprocessor_output'].get('fatal_error')}")

    if result["preprocessor_output"].get("fatal_error"):
        return {**state, "fatal_error": result["preprocessor_output"]["fatal_error"],
                "_preprocessor_response": result["final_response"]}

    cleaned_df = result["cleaned_df"]
    print(f"[NODE: preprocessor_run] cleaned_df shape: {cleaned_df.shape if cleaned_df is not None else 'None'}")

    preprocessor_serialized = _serialize_output(result["preprocessor_output"])
    print(f"[NODE: preprocessor_run] preprocessor_serialized keys: {list(preprocessor_serialized.keys())}")

    return {
        **state,
        "cleaned_df":             _df_to_state(cleaned_df),
        "preprocessor_output":    preprocessor_serialized,
        "_preprocessor_response": result["final_response"],
    }


def node_preprocessor_confirm(state: AristostatState) -> AristostatState:
    """Shows cleaning summary to user and confirms — contains the interrupt."""
    print("\n[NODE: preprocessor_confirm] Starting...")

    if state.get("fatal_error"):
        interrupt({"message": state.get("_preprocessor_response", ""), "type": "fatal_error"})
        return state

    user_response = interrupt({
        "message": state.get("_preprocessor_response", ""),
        "prompt":  "Shall I proceed with the cleaned data? (yes/no)",
        "type":    "confirm",
    })

    if str(user_response).strip().lower() not in ("yes", "y"):
        print("[NODE: preprocessor_confirm] User stopped pipeline.")
        return {**state, "fatal_error": "User stopped pipeline after preprocessing."}

    return state


# ── ASSUMPTION CHECKER ──

def node_assumption_checker_run(state: AristostatState) -> AristostatState:
    """Runs assumption checker — no interrupts."""
    print("\n[NODE: assumption_checker_run] Starting...")
    print(f"[NODE: assumption_checker_run] methodologist_output: {state.get('methodologist_output')}")
    

    rectified = _df_from_state(state.get("rectified_df"))
    cleaned   = _df_from_state(state["cleaned_df"])
    df = rectified if rectified is not None else cleaned
    print(f"[NODE: assumption_checker_run] df shape: {df.shape if df is not None else 'None'}")
    print(f"[NODE: assumption_checker_run] using rectified_df: {rectified is not None}")

    result = run_assumption_checker(
        methodologist_output=state["methodologist_output"],
        cleaned_df=df,
        profiler_output=state["profiler_output"],
    )

    checker_output = result["checker_output"]
    print(f"[NODE: assumption_checker_run] has_failures: {checker_output.get('has_failures')}")
    print(f"[NODE: assumption_checker_run] results: {checker_output.get('results')}")

    checker_serialized = _serialize_output(checker_output)

    return {
        **state,
        "checker_output":   checker_serialized,
        "_checker_response": result["final_response"],
    }


def node_assumption_checker_confirm(state: AristostatState) -> AristostatState:
    """Shows assumption results to user — contains the interrupt."""
    print("\n[NODE: assumption_checker_confirm] Starting...")

    checker = state.get("checker_output", {})
    print(f"[NODE: assumption_checker_confirm] has_failures: {checker.get('has_failures')}")

    if not checker.get("has_failures"):
        interrupt({
            "message": state.get("_checker_response", ""),
            "prompt":  "All assumptions met. Proceeding to run the test.",
            "type":    "info",
        })
        return {**state, "user_rectify_or_proceed": "proceed"}

    user_response = interrupt({
        "message": state.get("_checker_response", ""),
        "prompt":  "Would you like to rectify the failed assumptions or proceed anyway? (rectify/proceed)",
        "type":    "rectify_or_proceed",
    })

    decision = str(user_response).strip().lower()
    print(f"[NODE: assumption_checker_confirm] user decision: {decision}")

    return {
        **state,
        "user_rectify_or_proceed": "rectify" if "rectify" in decision else "proceed",
    }


# ── RECTIFICATION STRATEGIST ──

def node_rectification_strategist_run(state: AristostatState) -> AristostatState:
    """Runs rectification strategist LLM — no interrupts."""
    print("\n[NODE: rectification_strategist_run] Starting...")
    

    attempt = state.get("rectification_attempt", 1)
    phase   = state.get("rectification_phase", "pre_test")
    print(f"[NODE: rectification_strategist_run] attempt: {attempt}, phase: {phase}")

    rectified = _df_from_state(state.get("rectified_df"))
    cleaned   = _df_from_state(state["cleaned_df"])
    df = rectified if rectified is not None else cleaned
    print(f"[NODE: rectification_strategist_run] df shape: {df.shape if df is not None else 'None'}")

    checker = state.get("checker_output", {})
    critic  = state.get("critic_output", {})

    if phase == "pre_test":
        failed = [r["name"] for r in checker.get("results", []) if r.get("status") == "failed"]
    else:
        failed = [r["name"] for r in critic.get("results", []) if r.get("status") == "failed"]
    print(f"[NODE: rectification_strategist_run] failed assumptions: {failed}")

    result = run_rectification_strategist(
        failed_assumptions=failed,
        phase=phase,
        cleaned_df=df,
        methodologist_output=state["methodologist_output"],
        rectification_attempt=attempt,
        max_attempts=3,
    )
    print(f"[DEBUG] raw result rectification_output type: {type(result['rectification_output'])}")
    print(f"[DEBUG] raw proposed_solutions: {result['rectification_output'].get('proposed_solutions')}")
    rect_output = _serialize_output(result["rectification_output"])
    print(f"[DEBUG run] serialized proposed_solutions: {rect_output.get('proposed_solutions')}")
    print(f"[NODE: rectification_strategist_run] proposed_solutions: {len(rect_output.get('proposed_solutions', []))}")

    return {
        **state,
        "rectification_output":    rect_output,
        "_rectification_response": result["final_response"],
        "_rectification_failed":   failed,
        "_rectification_df_json":  _df_to_state(df),
    }


def node_rectification_strategist_confirm(state: AristostatState) -> AristostatState:
    """Shows solutions to user and applies chosen one — contains the interrupt."""
    print("\n[NODE: rectification_strategist_confirm] Starting...")

    attempt     = state.get("rectification_attempt", 1)
    phase       = state.get("rectification_phase", "pre_test")
    rect_output = state.get("rectification_output", {})
    failed      = state.get("_rectification_failed", [])
    df          = _df_from_state(state.get("_rectification_df_json"))

    user_response = interrupt({
        "message": state.get("_rectification_response", ""),
        "prompt":  "Please choose an option (enter the number, or type 'proceed' to continue despite failures).",
        "type":    "choose_solution",
        "options": rect_output.get("proposed_solutions", []),
    })
    print(f"[NODE: rectification_strategist_confirm] user response: {user_response}")

    if str(user_response).strip().lower() == "proceed":
        from Tools.rectification_strategist import (
            init_rectification_store, get_rectification_store,
            accept_violation_and_proceed,
        )
        init_rectification_store(
            failed_assumptions=failed,
            phase=phase,
            cleaned_df=df,
            methodologist_output=state["methodologist_output"],
            rectification_attempt=attempt,
        )
        accept_violation_and_proceed.invoke({"violation_names": ",".join(failed)})
        rect_output = _serialize_output(get_rectification_store()["rectification_output"].model_dump())
        print("[NODE: rectification_strategist_confirm] User accepted violation.")
        return {
            **state,
            "rectification_output":    rect_output,
            "rectified_df":            _df_to_state(df),
            "user_rectify_or_proceed": "proceed",
        }

    solution_id = _resolve_solution_choice(user_response, rect_output.get("proposed_solutions", []))
    # ── Special case: drop variable — ask which one ──
    if solution_id == "multicollinearity_drop_variable":
        ind_vars = state["methodologist_output"].get("independent_variables", [])

        # Find high-VIF variables from checker results to suggest to user
        high_vif_vars = _get_high_vif_variables(state.get("checker_output", {}))
        suggestion = f" (suggested: {', '.join(high_vif_vars)})" if high_vif_vars else ""

        drop_response = interrupt({
            "message": f"Which variable would you like to drop?{suggestion}\nAvailable predictors: {', '.join(ind_vars)}",
            "prompt":  "Enter the exact variable name to drop: ",
            "type":    "confirm",
        })

        column_to_drop = str(drop_response).strip()

        # Inject the chosen column into the solution details
        rect_output_modified = dict(rect_output)
        for sol in rect_output_modified.get("proposed_solutions", []):
            if isinstance(sol, dict) and sol.get("solution_id") == "multicollinearity_drop_variable":
                sol["action_details"] = {"column": column_to_drop}
            elif hasattr(sol, "solution_id") and sol.solution_id == "multicollinearity_drop_variable":
                sol.action_details["column"] = column_to_drop

        # Also update methodologist_output to remove the dropped variable
        updated_ind_vars = [v for v in ind_vars if v != column_to_drop]
        updated_methodologist = {
            **state["methodologist_output"],
            "independent_variables": updated_ind_vars,
        }
    print(f"[NODE: rectification_strategist_confirm] chosen solution_id: {solution_id}")
    print(f"[DEBUG] proposed_solutions: {rect_output.get('proposed_solutions')}")
    print(f"[DEBUG] user_response raw: {repr(user_response)}")
    solution_id = _resolve_solution_choice(user_response, rect_output.get("proposed_solutions", []))
    print(f"[DEBUG] resolved solution_id: {solution_id}")
    from Tools.rectification_strategist import init_rectification_store, get_rectification_store
    from core.rectification_engine import build_rectification_output
    from Schemas.rectification_strategist import RectificationPhase

    ph = RectificationPhase(phase)
    rectified_df, applied_output = build_rectification_output(
        failed_assumptions=failed,
        phase=ph,
        chosen_solution_id=solution_id,
        df=df,
        dependent_var=state["methodologist_output"].get("dependent_variable"),
        independent_vars=state["methodologist_output"].get("independent_variables", []),
        rectification_attempt=attempt,
        max_attempts=3,
    )
    print(f"[NODE: rectification_strategist_confirm] rectified_df shape: {rectified_df.shape if rectified_df is not None else 'None'}")
    print(f"[NODE: rectification_strategist_confirm] new_test: {applied_output.new_test}")

    rect_serialized = _serialize_output(applied_output)
    updated_methodologist = state["methodologist_output"]
    if applied_output.new_test:
        updated_methodologist = {
            **state["methodologist_output"],
            "selected_test": applied_output.new_test,
        }

    return {
        **state,
        "rectification_output":    rect_serialized,
        "rectified_df":            _df_to_state(rectified_df),
        "methodologist_output":    updated_methodologist,
        "rectification_attempt":   attempt + 1,
        "user_rectify_or_proceed": "rectified",
    }


# ── STATISTICIAN ──

def node_statistician(state: AristostatState) -> AristostatState:
    """Executes the statistical test."""
    print("\n[NODE: statistician] Starting...")
    print(f"[NODE: statistician] methodologist_output: {state.get('methodologist_output')}")
    

    rectified = _df_from_state(state.get("rectified_df"))
    cleaned   = _df_from_state(state["cleaned_df"])
    df = rectified if rectified is not None else cleaned
    print(f"[NODE: statistician] df shape: {df.shape if df is not None else 'None'}")
    print(f"[NODE: statistician] using rectified_df: {rectified is not None}")

    result = run_statistician(
        methodologist_output=state["methodologist_output"],
        cleaned_df=df,
        rectification_output=state.get("rectification_output"),
    )
    print(f"[NODE: statistician] result keys: {list(result.keys())}")
    statistician_serialized = _serialize_output(result["statistician_output"])
    print(f"[NODE: statistician] statistician_output keys: {list(statistician_serialized.keys())}")

    return {**state, "statistician_output": statistician_serialized}


# ── MODEL CRITIC ──

def node_model_critic_run(state: AristostatState) -> AristostatState:
    """Runs model critic — no interrupts."""
    print("\n[NODE: model_critic_run] Starting...")
    print(f"[NODE: model_critic_run] statistician_output keys: {list(state.get('statistician_output', {}).keys())}")
    

    rectified = _df_from_state(state.get("rectified_df"))
    cleaned   = _df_from_state(state["cleaned_df"])
    df = rectified if rectified is not None else cleaned
    print(f"[NODE: model_critic_run] df shape: {df.shape if df is not None else 'None'}")

    result = run_model_critic(
        statistician_output=state["statistician_output"],
        fitted_model=get_fitted_model(),
        cleaned_df=df,
        methodologist_output=state["methodologist_output"],
    )

    critic_output = result["critic_output"]
    print(f"[NODE: model_critic_run] checks_applicable: {critic_output.get('checks_applicable')}")
    print(f"[NODE: model_critic_run] has_failures: {critic_output.get('has_failures')}")

    critic_serialized = _serialize_output(critic_output)

    return {
        **state,
        "critic_output":    critic_serialized,
        "_critic_response": result["final_response"],
    }


def node_model_critic_confirm(state: AristostatState) -> AristostatState:
    """Shows critic results to user — contains the interrupt."""
    print("\n[NODE: model_critic_confirm] Starting...")

    critic = state.get("critic_output", {})
    print(f"[NODE: model_critic_confirm] checks_applicable: {critic.get('checks_applicable')}")
    print(f"[NODE: model_critic_confirm] has_failures: {critic.get('has_failures')}")

    if not critic.get("checks_applicable", False):
        print("[NODE: model_critic_confirm] Non-regression — skipping.")
        return state

    if not critic.get("has_failures"):
        interrupt({
            "message": state.get("_critic_response", ""),
            "prompt":  "All post-test checks passed. Proceeding to final report.",
            "type":    "info",
        })
        return {**state, "user_rectify_or_proceed": "proceed"}

    user_response = interrupt({
        "message": state.get("_critic_response", ""),
        "prompt":  "Would you like to rectify the post-test failures or proceed anyway? (rectify/proceed)",
        "type":    "rectify_or_proceed",
    })

    decision = str(user_response).strip().lower()
    print(f"[NODE: model_critic_confirm] user decision: {decision}")

    return {
        **state,
        "rectification_phase":     "post_test",
        "user_rectify_or_proceed": "rectify" if "rectify" in decision else "proceed",
    }


# ── FINAL REPORT ──

def node_final_report(state: AristostatState) -> AristostatState:
    """Generates the final report."""
    print("\n[NODE: final_report] Starting...")
    print(f"[NODE: final_report] statistician_output keys: {list(state.get('statistician_output', {}).keys())}")
    print(f"[NODE: final_report] checker_output keys: {list(state.get('checker_output', {}).keys())}")
    print(f"[NODE: final_report] critic_output present: {state.get('critic_output') is not None}")
    print(f"[NODE: final_report] rectification_output present: {state.get('rectification_output') is not None}")
    
    try:
        result = run_final_report(
            original_query=state["user_query"],
            profiler_output=state["profiler_output"],
            preprocessor_output=state["preprocessor_output"],
            methodologist_output=state["methodologist_output"],
            checker_output=state["checker_output"],
            statistician_output=state["statistician_output"],
            rectification_output=state.get("rectification_output"),
            critic_output=state.get("critic_output"),
        )

        print(f"[NODE: final_report] report_output keys: {list(result.get('report_output', {}).keys())}")
        report_serialized = _serialize_output(result["report_output"])
        # ── Final report display ──
        print(f"[NODE: final_report] result keys: {list(result.keys())}")
        print(f"[NODE: final_report] report_output: {result.get('report_output')}")
        print(f"[NODE: final_report] report_serialized keys: {list(report_serialized.keys())}")
        return {**state, "report_output": report_serialized}

    except Exception as e:
        import traceback
        print(f"[NODE: final_report] ERROR: {str(e)}")
        print(traceback.format_exc())
    return {**state, "report_output": {}, "fatal_error": str(e)}


# ─────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# ─────────────────────────────────────────────

def route_after_profiler(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "intent_interpreter_run"


def route_after_intent(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "methodologist_run"


def route_after_methodologist(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "preprocessor_run"


def route_after_preprocessor(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "assumption_checker_run"


def route_after_assumption_checker(state: AristostatState) -> str:
    checker  = state.get("checker_output", {})
    decision = state.get("user_rectify_or_proceed", "proceed")
    if not checker.get("has_failures"):
        return "statistician"
    if decision == "rectify":
        return "rectification_strategist_run"
    return "statistician"


def route_after_rectification(state: AristostatState) -> str:
    rect    = state.get("rectification_output", {})
    attempt = state.get("rectification_attempt", 1)
    if attempt > 3:
        return "statistician"
    if rect.get("user_accepted_violation"):
        return "statistician"
    if state.get("rectification_phase") == "post_test":
        return "statistician"
    next_step = rect.get("next_step", "assumption_checker_run")
    if next_step == "statistician":
        return "statistician"
    return "assumption_checker_run"


def route_after_statistician(state: AristostatState) -> str:
    return "model_critic_run"


def route_after_model_critic(state: AristostatState) -> str:
    critic   = state.get("critic_output", {})
    decision = state.get("user_rectify_or_proceed", "proceed")
    if not critic.get("checks_applicable", False):
        return "final_report"
    if not critic.get("has_failures"):
        return "final_report"
    if decision == "rectify":
        return "rectification_strategist_run"
    return "final_report"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _resolve_combination_choice(user_response: Any, combinations: list) -> str:
    try:
        idx  = int(str(user_response).strip()) - 1
        combo = combinations[idx]
        cols  = combo.get("columns", [])
        goal  = combo.get("suggested_goal", "")
        return f"Please run {goal} analysis on {' and '.join(cols)}"
    except (ValueError, IndexError):
        return str(user_response)


def _resolve_solution_choice(user_response: Any, solutions: list) -> str | None:
    response = str(user_response).strip()
    try:
        idx = int(response) - 1
        if 0 <= idx < len(solutions):
            sol = solutions[idx]
            return sol.get("solution_id") if isinstance(sol, dict) else sol.solution_id
    except ValueError:
        return response
    return None

def _get_high_vif_variables(checker_output: dict) -> list[str]:
    """Extract variable names with high VIF from checker results for user suggestion."""
    for result in checker_output.get("results", []):
        if "multicollinearity" in result.get("name", "").lower() or \
           "vif" in result.get("plain_reason", "").lower():
            reason = result.get("plain_reason", "")
            # Parse "VIF scores: {Age: 27.14 (≥10), Years of Experience: 27.24 (≥10), ...}"
            import re
            high_vif = re.findall(r"([\w\s]+?):\s*[\d.]+\s*\(≥10\)", reason)
            return [v.strip() for v in high_vif]
    return []

# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AristostatState)

    # ── Register nodes ──
    builder.add_node("data_profiler",                       node_data_profiler)
    builder.add_node("intent_interpreter_run",              node_intent_interpreter_run)
    builder.add_node("intent_interpreter_confirm",          node_intent_interpreter_confirm)
    builder.add_node("methodologist_run",                   node_methodologist_run)
    builder.add_node("methodologist_confirm",               node_methodologist_confirm)
    builder.add_node("preprocessor_run",                    node_preprocessor_run)
    builder.add_node("preprocessor_confirm",                node_preprocessor_confirm)
    builder.add_node("assumption_checker_run",              node_assumption_checker_run)
    builder.add_node("assumption_checker_confirm",          node_assumption_checker_confirm)
    builder.add_node("rectification_strategist_run",        node_rectification_strategist_run)
    builder.add_node("rectification_strategist_confirm",    node_rectification_strategist_confirm)
    builder.add_node("statistician",                        node_statistician)
    builder.add_node("model_critic_run",                    node_model_critic_run)
    builder.add_node("model_critic_confirm",                node_model_critic_confirm)
    builder.add_node("final_report",                        node_final_report)

    # ── Entry point ──
    builder.set_entry_point("data_profiler")

    # ── Edges ──
    builder.add_conditional_edges("data_profiler",                    route_after_profiler)
    builder.add_edge("intent_interpreter_run",                        "intent_interpreter_confirm")
    builder.add_conditional_edges("intent_interpreter_confirm",       route_after_intent)
    builder.add_edge("methodologist_run",                             "methodologist_confirm")
    builder.add_conditional_edges("methodologist_confirm",            route_after_methodologist)
    builder.add_edge("preprocessor_run",                              "preprocessor_confirm")
    builder.add_conditional_edges("preprocessor_confirm",             route_after_preprocessor)
    builder.add_edge("assumption_checker_run",                        "assumption_checker_confirm")
    builder.add_conditional_edges("assumption_checker_confirm",       route_after_assumption_checker)
    builder.add_edge("rectification_strategist_run",                  "rectification_strategist_confirm")
    builder.add_conditional_edges("rectification_strategist_confirm", route_after_rectification)
    builder.add_conditional_edges("statistician",                     route_after_statistician)
    builder.add_edge("model_critic_run",                              "model_critic_confirm")
    builder.add_conditional_edges("model_critic_confirm",             route_after_model_critic)
    builder.add_edge("final_report",                                  END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# def save_graph_image(output_path: str = "aristostat_graph.png") -> None:
#     try:
#         png_bytes = graph.get_graph().draw_mermaid_png()
#         with open(output_path, "wb") as f:
#             f.write(png_bytes)
#         print(f"Graph diagram saved to: {output_path}")
#     except Exception as e:
#         print(f"Could not render PNG ({e}). Mermaid source:\n")
#         print(graph.get_graph().draw_mermaid())


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

graph = build_graph()


def run_aristostat(
    csv_path: str,
    user_query: str,
    thread_id: str = "default",
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AristostatState = {
        "csv_path":                csv_path,
        "user_query":              user_query,
        "rectification_attempt":   1,
        "rectification_phase":     "pre_test",
        "fatal_error":             None,
        "rectification_output":    None,
        "critic_output":           None,
        "rectified_df":            None,
        "user_rectify_or_proceed": "proceed",
    }
    return graph.invoke(initial_state, config=config)


def resume_aristostat(
    user_response: Any,
    thread_id: str = "default",
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=user_response), config=config)


def _get_interrupt(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
    return None


# ─────────────────────────────────────────────
# CLI RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python main.py <csv_path> <query>")
        sys.exit(1)

    csv_path   = sys.argv[1]
    user_query = sys.argv[2]
    thread_id  = "cli_session_1"

    print("\n" + "=" * 60)
    print("  ARISTOSTAT: THE LOGICAL INFERENCE ENGINE")
    print("=" * 60)

    state = run_aristostat(csv_path, user_query, thread_id)

    while True:
        interrupt_value = _get_interrupt(state)

        if interrupt_value is None:
            break

        interrupt_type = interrupt_value.get("type", "confirm")
        print(f"\n[AGENT]: {interrupt_value.get('message', '')}")

        if interrupt_type == "fatal_error":
            print("\n[!!!] FATAL ERROR: Pipeline cannot continue.")
            sys.exit(1)

        if interrupt_type in ["choose_combination", "choose_solution"] and interrupt_value.get("options"):
            for i, opt in enumerate(interrupt_value["options"], 1):
                if isinstance(opt, dict):
                    desc    = opt.get("description") or opt.get("suggested_goal") or str(opt)
                    cols    = opt.get("columns", [])
                    col_str = f" ({', '.join(cols)})" if cols else ""
                    print(f"  {i}. {desc}{col_str}")
                else:
                    print(f"  {i}. {opt}")

        if interrupt_type == "info":
            print("[Auto-proceeding...]")
            user_input = "yes"
        else:
            prompt     = interrupt_value.get("prompt", "Your response: ")
            user_input = input(f"\n{prompt} ").strip()

        state = resume_aristostat(user_input, thread_id)

    if state:
        report = state.get("report_output", {})
        if report and report.get("docx_path"):
            print(f"\nSUCCESS: Analysis complete.")
            print(f"Report Location: {report.get('docx_path')}")
        if state.get("fatal_error"):
            print(f"\nPIPELINE STOPPED: {state['fatal_error']}")