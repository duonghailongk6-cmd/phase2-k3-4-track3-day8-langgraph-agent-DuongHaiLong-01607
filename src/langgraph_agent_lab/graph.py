"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow.

    Build the complete graph with this architecture:

    START → intake → classify → [conditional: route_after_classify]
      simple       → answer → finalize → END
      tool         → tool → evaluate → [conditional: route_after_evaluate]
                                          success → answer → finalize → END
                                          needs_retry → retry → [conditional: route_after_retry]
                                                                  tool (retry)
                                                                  dead_letter → finalize → END
      missing_info → clarify → finalize → END
      risky        → risky_action → approval → [conditional: route_after_approval]
                                                  approved → tool → evaluate → ...
                                                  rejected → clarify → finalize → END
      error        → retry → [conditional: route_after_retry] → ...

    Reference: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
    """
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
      answer_node,
      approval_node,
      ask_clarification_node,
      classify_node,
      dead_letter_node,
      evaluate_node,
      finalize_node,
      intake_node,
      retry_or_fallback_node,
      risky_action_node,
      tool_node,
    )
    from .routing import (
      route_after_approval,
      route_after_classify,
      route_after_evaluate,
      route_after_retry,
    )

    workflow = StateGraph(AgentState)
    for name, node in {
      "intake": intake_node,
      "classify": classify_node,
      "tool": tool_node,
      "evaluate": evaluate_node,
      "answer": answer_node,
      "clarify": ask_clarification_node,
      "risky_action": risky_action_node,
      "approval": approval_node,
      "retry": retry_or_fallback_node,
      "dead_letter": dead_letter_node,
      "finalize": finalize_node,
    }.items():
      workflow.add_node(name, node)
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "classify")
    workflow.add_conditional_edges("classify", route_after_classify)
    workflow.add_edge("tool", "evaluate")
    workflow.add_conditional_edges("evaluate", route_after_evaluate)
    workflow.add_conditional_edges("retry", route_after_retry)
    workflow.add_edge("dead_letter", "finalize")
    workflow.add_edge("answer", "finalize")
    workflow.add_edge("clarify", "finalize")
    workflow.add_edge("risky_action", "approval")
    workflow.add_conditional_edges("approval", route_after_approval)
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)
