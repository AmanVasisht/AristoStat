# Aristostat — The Logical Inference Engine

> An agentic AI system that automates end-to-end statistical analysis: from raw data ingestion through test selection, assumption validation, violation rectification, and professional report generation — with human-in-the-loop checkpoints throughout.

---

## Overview

Aristostat is a multi-agent LangGraph pipeline that acts as an intelligent statistical analyst. Given a dataset and a research question in plain English, it selects the appropriate statistical test, rigorously checks all required assumptions, proposes and applies rectifications when violations are detected, runs the test, critiques the model, and generates a structured report — pausing at key decision points to incorporate user judgment.

The system is designed around a core principle: **statistical reasoning should be transparent, explainable, and collaborative**, not a black box.

---
## 🎥 Aristostat 5-Minute UI Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=qxqWue2mDQc" target="_blank">
    <img src="assets/Aristostat_disclaimer.png" alt="Aristostat Demo" width="800">
  </a>
</p>

Click the image above to watch the full demo on YouTube.

---

## Architecture

The pipeline is orchestrated as a **directed cyclic graph** using LangGraph's `StateGraph`, with nine specialised agents connected by conditional edges and human interrupt checkpoints.

```
Data Profiler [interrupt: confirm proceed]
     ↓
Intent Interpreter Run → Intent Interpreter Confirm [interrupt: confirm / correct]
     ↓
Methodologist Run → Methodologist Confirm [interrupt: confirm / correct test]
     ↓  ↑ (rerun loop if user corrects test)
Preprocessor Run → Preprocessor Confirm [interrupt: confirm cleaned data]
     ↓
Assumption Checker Run → Assumption Checker Confirm [interrupt: rectify or proceed?]
     ↓ pass                        ↓ rectify
     │              Rectification Strategist Run
     │                          ↓
     │              Rectification Strategist Confirm [interrupt: choose solution]
     │                          ↓
     │         ┌────────────────┴──────────────────┐
     │    assumption_checker_run              statistician
     │    (re-check after fix,           (test switch / drop /
     │     max 3 attempts)                proceed accepted)
     ↓                                          ↓
Statistician ◄────────────────────────────────-┘
     ↓
Model Critic Run → Model Critic Confirm [interrupt: rectify or proceed? — regression only]
     ↓ pass / non-regression          ↓ rectify
     │                   Rectification Strategist Run
     │                          ↓
     │              Rectification Strategist Confirm
     │                          ↓
     │                      Statistician (re-run)
     ↓
Final Report → END
```

### Agent Design: ReAct vs Direct Calling

Agents are split into two architectural patterns based on their cognitive requirements:

| Pattern | Agents | Rationale |
|---|---|---|
| **ReAct (Tool-calling loop)** | Intent Interpreter, Methodologist, Assumption Checker, Rectification Strategist, Final Report | Require multi-step reasoning, tool invocation, and iterative decision-making. The ReAct loop allows the agent to gather context, reason about it, and act — rather than making a single-shot inference. |
| **Direct calling** |  Data Profiler, Preprocessor, Statistician, Model Critic | Execute deterministic or well-scoped tasks where tool-use overhead adds no value. Faster, cheaper, and more reliable for structured computation. |

---

## Agent Responsibilities

| Agent | Role |
|---|---|
| **Data Profiler** | Loads and profiles the dataset — column types, missingness, cardinality, distributions |
| **Intent Interpreter** | Parses the user's natural language query into structured analytical intent |
| **Methodologist** | Selects the statistically appropriate test given intent and data characteristics |
| **Preprocessor** | Cleans data — handles missing values, type coercion, encoding |
| **Assumption Checker** | Runs all pre-test statistical assumptions relevant to the chosen test |
| **Rectification Strategist** | Proposes and applies solutions when assumptions are violated (test switching, variable dropping, transformations) |
| **Statistician** | Executes the statistical test and returns structured results |
| **Model Critic** | Runs post-test diagnostics — residual normality, homoscedasticity, autocorrelation, influential points |
| **Final Report** | Synthesises all pipeline outputs into a structured Word report |

---

## Supported Statistical Tests

| Family | Tests |
|---|---|
| **Inference** | Independent T-Test, Paired T-Test, One-Sample T-Test, One-Way ANOVA, Two-Way ANOVA, Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis |
| **Regression** | Simple Linear Regression, Multiple Linear Regression, Ridge Regression, Lasso Regression (with cross-validated alpha) |
| **Correlation** | Pearson, Spearman |
| **Dimensionality** | Principal Component Analysis |

---

## Key Technical Design Decisions

**LangGraph StateGraph with MemorySaver checkpointing** — the full pipeline state (dataframes serialised as JSON, agent outputs, routing flags) persists across human interrupts, enabling the pipeline to pause, collect input, and resume without losing context.

**LLM-powered semantic resolution** — when a user types "drop age" instead of selecting a numbered option, a focused LLM call maps the free text to the exact column name using the actual dataset's column list. No hardcoded abbreviation tables.

**Schema-first outputs** — every agent produces a typed Pydantic schema (`MethodologistOutput`, `StatisticianOutput`, etc.). This enforces contract boundaries between agents and prevents silent data loss from field mismatches.

**Sklearn pipeline feature name tracking** — Ridge and Lasso models store encoded feature column order as a custom attribute (`_aristostat_feature_names`) on the fitted pipeline, enabling the Model Critic to correctly reconstruct inputs for residual extraction without re-encoding ambiguity.

**Assumption registry pattern** — pre-test checks are driven by a declarative `ASSUMPTION_REGISTRY` mapping test names to required checks with their function names and parameters. Adding a new test requires only a registry entry, not new dispatch logic.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) — StateGraph with conditional edges and interrupt checkpoints |
| **LLM Framework** | [LangChain](https://github.com/langchain-ai/langchain) |
| **LLM Inference** | [Groq](https://groq.com/) via `langchain-groq` (`ChatGroq`) — ultra-low latency inference |
| **Statistical computation** | `scipy`, `statsmodels`, `scikit-learn`, `pandas`, `numpy` |
| **Report generation** | `python-docx` |
| **API layer** | [FastAPI](https://fastapi.tiangolo.com/) with Server-Sent Events for real-time streaming |
| **Frontend** | Vanilla HTML/CSS/JS — single-file, no build step |
| **Schemas** | [Pydantic v2](https://docs.pydantic.dev/) |

### Models

Different agents are assigned models based on their reasoning complexity and latency requirements:

| Model | Used For |
|---|---|
| `meta-llama/llama-4-maverick-17b-128e-instruct` | High-stakes reasoning — Final Report |
| `qwen/qwen3-32b` | Semantic resolution tasks — intent parsing, column mapping |
| `llama-3.1-8b-instant` | Fast, low-cost tasks — intent classification, light reasoning |

---

## Project Structure

```
aristostat/
├── Agents/                  # Agent entry points (thin wrappers over core engines)
│   ├── data_profiler.py
│   ├── intent_interpreter.py
│   ├── methodologist.py
│   ├── preprocessor.py
│   ├── assumption_checker.py
│   ├── rectification_strategist.py
│   ├── statistician.py
│   ├── critic.py
│   └── final_report.py
├── core/                    # Pure statistical and reasoning engines (no LLM)
│   ├── assumption_engine.py
│   ├── statistician_engine.py
│   ├── rectification_engine.py
│   └── intent_engine.py
├── Schemas/                 # Pydantic output schemas for all agents
├── Tools/                   # LangChain tools used by ReAct agents
├── Utils/                   # Assumption registry, helpers
├── constants/               # Thresholds, configuration
├── Output/                  # Generated reports
├── main.py                  # LangGraph graph definition and CLI runner
├── api.py                   # FastAPI backend with SSE streaming
└── index.html               # Frontend UI
```

---

## Running the Project

### CLI

```bash
python main.py Salary_Data.csv "study the effect of all variables except job title on salary"
```

### Web UI

```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`, upload a CSV, and describe your analysis goal.

> **Note:** The web interface has been locally tested. Cloud deployment is part of the planned roadmap.

### API Key

All LLM calls route through the Groq API. Set your key before running:

```bash
export GROQ_API_KEY=your_key_here   # Linux / macOS
set GROQ_API_KEY=your_key_here      # Windows
```

---

## Status

The core pipeline is functional across the currently supported test suite. Active areas for further development include expanding the test library, improving assumption check coverage for multi-factor designs, and additional rectification strategies for post-test violations.

---

## Roadmap

- **Expanded test library** — survival analysis, mixed-effects models, time series tests, non-parametric alternatives for multi-factor designs
- **Graph complexity** — handling multi-outcome analyses, nested pipelines, and composite research questions in a single run
- **Conversational history** — Add reducers especially for SQL agent or rectification agent, as this can save history of their loops and enhance the final report agent output.
- **Code optimisation** — parallel assumption checks, caching profiler output across sessions, reducing redundant LLM calls
- **Deployment** — containerisation with Docker, CI/CD pipeline, cloud hosting (AWS / GCP), replacement of MemorySaver checkpoint with PostgresSaver.
- **Guardrails** — input validation, hallucination detection on statistical interpretation, confidence scoring on test selection
- **Visualisation** — inline chart generation (residual plots, Q-Q plots, coefficient plots) embedded in the report
- **Multi-dataset support** — comparative analyses across datasets, longitudinal study designs


⚠️ Note: This project is still actively being developed. There may be statistical inaccuracies present in certain scenarios; a more rigorous and in-depth implementation of statistical methods is currently in progress. Feedback and suggestions are always welcome!
---

*Built with LangGraph · ChatGroq · FastAPI*