"""
Deterministic graders for all three tasks.
Each returns a score strictly in (0.0, 1.0).
"""

from typing import Any, Dict, List


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return round(max(0.001, min(0.999, score)), 4)


def grade_easy(action_log: List[str], final_state: Dict[str, Any]) -> float:
    """
    Easy: CPU spike on api service.
    Optimal: scale_service(api)
    """
    score = 0.0

    if "scale_service" in action_log:
        score += 0.5

    api_status = final_state.get("system_snapshot", {}).get("api", {}).get("status", "")
    if api_status == "healthy":
        score += 0.3

    if "page_oncall" not in action_log:
        score += 0.1

    if "rollback_deploy" not in action_log:
        score += 0.1

    return _clamp(score)


def grade_medium(action_log: List[str], final_state: Dict[str, Any]) -> float:
    """
    Medium: Hidden OOM in db buried in noisy logs.
    Optimal: check_logs → restart_pod(db)
    """
    score = 0.0

    if "check_logs" in action_log:
        score += 0.25

        if "restart_pod" in action_log:
            if action_log.index("check_logs") < action_log.index("restart_pod"):
                score += 0.1

    if "restart_pod" in action_log:
        score += 0.35

    db_status = final_state.get("system_snapshot", {}).get("db", {}).get("status", "")
    if db_status == "healthy":
        score += 0.2

    unnecessary_restarts = sum(1 for a in action_log if a == "restart_pod")
    if unnecessary_restarts == 1:
        score += 0.1

    return _clamp(score)


def grade_hard(action_log: List[str], final_state: Dict[str, Any]) -> float:
    """
    Hard: Cascading failure from bad deploy.
    Optimal:
    check_metrics → check_logs → rollback_deploy → verify_recovery → mark_resolved
    """
    score = 0.0

    if "check_metrics" in action_log:
        score += 0.1

    if "check_logs" in action_log:
        score += 0.1

    if all(x in action_log for x in ["check_metrics", "check_logs", "rollback_deploy"]):
        if (
            action_log.index("check_metrics") < action_log.index("rollback_deploy")
            and action_log.index("check_logs") < action_log.index("rollback_deploy")
        ):
            score += 0.15

    if "rollback_deploy" in action_log:
        score += 0.3

    if "verify_recovery" in action_log:
        score += 0.15

    snapshot = final_state.get("system_snapshot", {})
    if snapshot:
        all_healthy = all(svc.get("status") == "healthy" for svc in snapshot.values())
        if all_healthy:
            score += 0.15

    if "restart_pod" in action_log and "rollback_deploy" not in action_log:
        score -= 0.1

    return _clamp(score)


def run_all_graders(env_instance) -> Dict[str, float]:
    """
    Run all tasks and return scores.
    """
    from env.environment import IncidentEnv
    from env.models import Action

    results = {}

    for scenario in ["easy", "medium", "hard"]:
        env = IncidentEnv(scenario=scenario, seed=42)
        env.reset()

        done = False
        while not done:
            if "check_logs" not in env._action_log:
                action = Action(action_type="check_logs")
            elif "check_metrics" not in env._action_log:
                action = Action(action_type="check_metrics")
            else:
                action = Action(action_type="mark_resolved")

            _, _, done, _ = env.step(action)

        state = env.state()
        log = state["action_log"]

        if scenario == "easy":
            results["easy"] = grade_easy(log, state)
        elif scenario == "medium":
            results["medium"] = grade_medium(log, state)
        else:
            results["hard"] = grade_hard(log, state)

    return results