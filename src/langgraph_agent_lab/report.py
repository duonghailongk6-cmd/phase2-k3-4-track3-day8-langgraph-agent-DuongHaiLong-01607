"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Render metrics, scenario results, architecture, failure analysis, and improvements.
    """
    rows = "\n".join(
        f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} | "
        f"{'PASS' if item.success else 'FAIL'} | {item.retry_count} | {item.nodes_visited} |"
        for item in metrics.scenario_metrics
    )
    return f"""# LangGraph Agent Lab Report

## Metrics

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.1%} |
| Average nodes visited | {metrics.avg_nodes_visited:.1f} |
| Total retries | {metrics.total_retries} |
| Total interrupts | {metrics.total_interrupts} |
| Resume success | {metrics.resume_success} |

## Scenario Results

| Scenario | Expected | Actual | Result | Retries | Nodes |
|---|---|---|---|---:|---:|
{rows}

## Architecture

The workflow uses a typed, serializable `AgentState`. Append-only reducers preserve messages,
tool results, errors, and audit events; scalar fields hold the current route, retry gate, approval,
and response. Intake normalizes the request, structured LLM classification selects a route, and
conditional edges dispatch to answering, clarification, tool evaluation, or approval. Every path
passes through `finalize` before `END`, and the checkpointer keys runs by `thread_id`.

## Failure Analysis

Transient tool failures are recorded and retried with a bounded attempt counter. Once the limit is
reached, the dead-letter path returns an escalation response rather than looping forever. Risky
requests can also be paused for approval; rejection routes to clarification instead of executing
the action.

## Improvement Plan

Use an LLM judge for richer tool-result evaluation, replace mock tools with authenticated adapters,
and add replay-based recovery tests against the SQLite checkpointer.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
