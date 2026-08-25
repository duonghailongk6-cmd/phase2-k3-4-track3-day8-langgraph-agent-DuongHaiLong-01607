"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from .llm import get_llm
from .state import AgentState, make_event


class Classification(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"]


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    prompt = (
        "Classify this support request into exactly one route. Priority is risky, tool, "
        "missing_info, error, simple. Risky means side effects such as refunds, deletion, "
        "cancellation, or sending email. Tool means an information lookup. missing_info is "
        "too vague to act on. error means a system failure, timeout, or crash. simple is a "
        "general question. Return structured output only.\n\nRequest: " + state["query"]
    )
    result = get_llm().with_structured_output(Classification).invoke(prompt)
    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "events": [make_event("classify", "completed", f"route={result.route}")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    if state.get("route") == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt}"
    else:
        result = f"Tool result for request: {state['query']}"
    return {"tool_results": [result], "events": [make_event("tool", "completed", result)]}


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest = (state.get("tool_results") or [""])[-1]
    evaluation = "needs_retry" if "ERROR" in latest else "success"
    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", evaluation)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    context = "\n".join(state.get("tool_results", [])) or "No tool was needed."
    prompt = (
        "Answer the support request helpfully and concisely. Ground the answer only in the "
        "request and supplied context; do not claim an action succeeded unless the context "
        "supports it.\nRequest: " + state["query"] + "\nContext: " + context
    )
    if state.get("approval") is not None:
        prompt += f"\nApproval decision: {state['approval']}"
    response = get_llm().invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": str(content),
        "events": [make_event("answer", "completed", "response generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    question = f"Could you provide more detail about what you need help with regarding: {state['query']}?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    action = f"Perform the requested potentially irreversible action: {state['query']}"
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "completed", "action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt({"proposed_action": state.get("proposed_action", "")})
        approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else bool(decision)
        comment = decision.get("comment", "") if isinstance(decision, dict) else ""
    else:
        approved, comment = True, "Automatically approved for lab execution"
    approval = {"approved": approved, "reviewer": "mock-reviewer", "comment": comment}
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", f"approved={approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    message = f"Retry scheduled after tool failure (attempt {attempt})"
    return {
        "attempt": attempt,
        "errors": [message],
        "events": [make_event("retry", "completed", message)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = "The request could not be completed after the allowed number of attempts. It has been escalated for review."
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "completed", "retry limit exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
