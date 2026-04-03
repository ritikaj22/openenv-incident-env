"""
Deterministic graders for all three tasks.
Each returns a score in [0.0, 1.0].
"""

from typing import Any, Dict, List


def grade_easy(action_log: List[str], final_state: Dict[str, Any]) -> float:
    """
    Easy: CPU spike on api service.
    Optimal: scale_service(api)
    Full marks if resolved correctly; partial marks for partial progress.
    """
    score = 0.0

    # Did agent scale the correct service?
    if "scale_service" in action_log:
        score += 0.5

    # Is api now healthy?
    api_status = final_state.get("system_snapshot", {}).get("api", {}).get("status", "")
    if api_status == "healthy":
        score += 0.3

    # Did agent resolve without paging oncall unnecessarily?
    if "page_oncall" not in action_log:
        score += 0.1

    # Did agent not rollback (unnecessary and harmful here)?
    if "rollback_deploy" not in action_log:
        score += 0.1

    return round(min(score, 1.0), 4)


def grade_medium(action_log: List[str], final_state: Dict[str, Any]) -> float:
    """
    Medium: Hidden OOM in db buried in noisy logs.
    Optimal: check_logs → restart_pod(db)
    """
    score = 0.0

    # Did agent check logs first (diagnostic step)?
    if "check_logs" in action_log:
        score += 0.25
        # Bonus if logs checked before restart (correct order)
        if action_log.index("check_logs") < action_log.index("restart_pod") \
                if "restart_pod" in action_log else False:
            score += 0.1

    # Did agent restart the right pod?
    if "restart_pod" in action_log:
        score += 0.35

    # Is db now healthy?
    db_status = final_state.get("system_snapshot", {}).get("db", {}).get("status", "")
    if db_status == "healthy":
        score += 0.2

    # Did agent avoid unnecessarily restarting other services?
    unnecessary_restarts = sum(
        1 for i, a in enumerate(action_log)
        if a == "restart_pod"
    )
    if unnecessary_restarts == 1:
        score += 0.1  # exactly one restart — surgical

    return round(min(score, 1.0), 4)


def grade_hard(action_log: List[str], final_state: Dict[str, Any]) -> float:
    """
    Hard: Cascading failure from bad deploy.
    Optimal: check_metrics → check_logs → rollback_deploy(auth) → verify_recovery → mark_resolved
    """
    score = 0.0

    # Diagnostic completeness
    if "check_metrics" in action_log:
        score += 0.1
    if "check_logs" in action_log:
        score += 0.1

    # Both diagnostics done before fix (correct order)
    if "check_metrics" in action_log and "check_logs" in action_log and "rollback_deploy" in action_log:
        metrics_i = action_log.index("check_metrics")
        logs_i = action_log.index("check_logs")
        rollback_i = action_log.index("rollback_deploy")
        if metrics_i < rollback_i and logs_i < rollback_i:
            score += 0.15  # correct diagnostic-first order

    # Correct fix: rollback
    if "rollback_deploy" in action_log:
        score += 0.3

    # Verified recovery before closing
    if "verify_recovery" in action_log:
        score += 0.15

    # Correct final resolution
    snapshot = final_state.get("system_snapshot", {})
    all_healthy = all(
        svc.get("status") == "healthy"
        for svc in snapshot.values()
    )
    if all_healthy:
        score += 0.15

    # Penalty: agent restarted pods (symptom fix, not root cause)
    if "restart_pod" in action_log and "rollback_deploy" not in action_log:
        score -= 0.1

    return round(min(max(score, 0.0), 1.0), 4)


def run_all_graders(env_instance) -> Dict[str, float]:
    """
    Run all 3 tasks and return scores dict.
    Used by validation and reporting.
    """
    from env.environment import IncidentEnv
    from env.models import Action

    results = {}
    for scenario in ["easy", "medium", "hard"]:
        env = IncidentEnv(scenario=scenario, seed=42)
        obs = env.reset()

        # Run a minimal baseline (just to get a grader-compatible state)
        done = False
        while not done:
            # Dumb baseline: always check_logs then mark_resolved
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
        elif scenario == "hard":
            results["hard"] = grade_hard(log, state)

    return results
