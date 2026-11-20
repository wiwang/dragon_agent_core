"""MLflow evaluation tracking.

The EvaluatorNode calls ``log_evaluation`` with the final state + score; each
call becomes an MLflow run with the route/iteration as params and quality and
grounding counts as metrics — giving you an eval dashboard over time. ``mlflow``
is an optional dependency (``pip install .[obs]``).
"""

from __future__ import annotations

from ..state import AgentState


class MlflowTracker:
    def __init__(self, tracking_uri: str, experiment: str = "mini-lang"):
        import mlflow  # lazy: optional dependency

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow

    def log_evaluation(self, state: AgentState, score: float) -> None:
        with self._mlflow.start_run(run_name=state.run_id or state.thread_id):
            self._mlflow.log_params(
                {
                    "route": state.route,
                    "thread_id": state.thread_id,
                    "iteration": state.iteration,
                }
            )
            self._mlflow.log_metrics(
                {
                    "quality": score,
                    "num_tools": len(state.tool_results),
                    "num_docs": len(state.retrieved_docs),
                }
            )
