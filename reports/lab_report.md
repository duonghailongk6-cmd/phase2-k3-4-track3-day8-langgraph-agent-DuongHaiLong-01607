# Day 08 LangGraph Agent Lab Report

## 1. Team / student

- Name: Duong Hai Long
- Repository: `phase2-k3-4-track3-day8-langgraph-agent`
- Date: 2026-08-25

## 2. Architecture

The workflow is a compiled LangGraph `StateGraph` whose nodes receive `AgentState`
and return partial state updates. The fixed path is `START -> intake -> classify`.
The classifier uses structured LLM output to select one of five routes:

- `simple -> answer -> finalize -> END`
- `tool -> evaluate -> answer -> finalize -> END`
- `missing_info -> clarify -> finalize -> END`
- `risky -> risky_action -> approval`; approval proceeds to the tool path when approved
  and to clarification when rejected
- `error -> retry`; retry either returns to `tool` or proceeds to `dead_letter` when the
  attempt limit is reached

All paths terminate at `finalize -> END`. Tool results are evaluated after every tool
call, and retry routing is bounded by `max_attempts`. The answer node uses an LLM and
grounds its prompt in the original request, tool results, and approval decision.

## 3. State schema

| Field | Reducer | Purpose |
|---|---|---|
| `query`, `route`, `risk_level` | overwrite | Current normalized request and classification |
| `attempt`, `max_attempts` | overwrite | Bounded retry control |
| `evaluation_result` | overwrite | Retry-loop gate after tool evaluation |
| `pending_question`, `proposed_action`, `approval` | overwrite | Clarification and approval workflow state |
| `final_answer` | overwrite | Current user-facing response |
| `messages` | append | Conversation and intake messages |
| `tool_results` | append | Results from each tool attempt |
| `errors` | append | Retry and failure history |
| `events` | append | Serializable audit events for each node visit |

The append-only fields use the `operator.add` reducer. Scalar fields are overwritten
so that the state contains the current workflow decision while preserving audit history.

## 4. Scenario results

The sample data contains seven scenarios. Their expected routing and workflow outcomes
are listed below. Runtime values should be refreshed by running
`make run-scenarios` with an LLM provider configured.

| Scenario | Expected | Actual | Success | Nodes | Retries | Interrupts | Approval required | Approval observed | Latency (ms) | Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `S01_simple` | `simple` | Pending | Pending | Pending | 0 | 0 | No | Pending | Pending | Pending |
| `S02_tool` | `tool` | Pending | Pending | Pending | 0 | 0 | No | Pending | Pending | Pending |
| `S03_missing` | `missing_info` | Pending | Pending | Pending | 0 | 0 | No | Pending | Pending | Pending |
| `S04_risky` | `risky` | Pending | Pending | Pending | 0 | 1 | Yes | Pending | Pending | Pending |
| `S05_error` | `error` | Pending | Pending | Pending | 1+ | 0 | No | Pending | Pending | Pending |
| `S06_delete` | `risky` | Pending | Pending | Pending | 0 | 1 | Yes | Pending | Pending | Pending |
| `S07_dead_letter` | `error` | Pending | Pending | Pending | 1 | 0 | No | Pending | Pending | Pending |

Each generated `scenario_metrics` object contains exactly the required fields:
`scenario_id`, `expected_route`, `actual_route`, `success`, `nodes_visited`,
`retry_count`, `interrupt_count`, `approval_required`, `approval_observed`,
`latency_ms`, and `errors`.

Automated unit validation currently reports **17 passed** tests. The six API-dependent
graph smoke cases are skipped when no LLM API key is configured. No `outputs/metrics.json`
was included because executing the scenario runner without a provider would not produce
valid runtime metrics.

## 5. Failure analysis

1. **Transient tool failure:** The tool node emits an `ERROR` result for error scenarios
   while the attempt count is below the configured limit. The evaluator routes that result
   to `retry`; the retry node increments `attempt`, records an error, and routes back to
   the tool while `attempt < max_attempts`. At the limit, the graph uses `dead_letter`
   and still passes through `finalize`, preventing an unbounded loop.

2. **Risky action without approval:** Refunds, deletions, cancellations, and email side
   effects are classified as risky. The graph prepares a proposed action and enters the
   approval node before calling the tool. A rejected decision routes to clarification,
   so the action is not executed without approval.

3. **Insufficient request detail:** Vague requests such as “Can you fix it?” route to
   clarification rather than causing the answer model to invent missing context.

## 6. Persistence / recovery evidence

`build_checkpointer()` supports `MemorySaver` for local execution and `SqliteSaver` for
the persistence extension. SQLite connections use `check_same_thread=False` and enable
WAL mode. Each run receives a stable `thread_id` derived from its scenario ID, and the
CLI passes that ID in the LangGraph configurable run context.

The current local verification uses the memory checkpointer. Crash-resume or state-history
replay has not been demonstrated, so `resume_success` should remain `false` until that
extension is executed and recorded in `outputs/metrics.json`.

## 7. Extension work

The SQLite checkpointer adapter is implemented, including WAL configuration and a clear
installation error when `langgraph-checkpoint-sqlite` is unavailable. Real HITL support
is also available through `LANGGRAPH_INTERRUPT=true`; the default remains mock approval
so unattended tests can run. Postgres, time travel, fan-out/fan-in, and tracing are not
implemented in this submission.

## 8. Improvement plan

The first production step would be to run the scenario suite against a configured model
and persist `outputs/metrics.json` plus a real `resume_success` demonstration. Next, I
would replace the mock tool with authenticated adapters, add an LLM-as-judge evaluator
with a strict structured schema, and add tests for approval rejection, SQLite recovery,
malformed model output, and provider failures. Finally, I would add latency capture to
events and make the CLI report provider and persistence configuration without exposing
secrets.