import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, Observation, Reward,
    ServiceState, ServiceStatus
)

VALID_ACTIONS = [
    "check_logs",
    "check_metrics",
    "scale_service",
    "restart_pod",
    "rollback_deploy",
    "page_oncall",
    "verify_recovery",
    "mark_resolved",
]


class IncidentEnv:
    """
    AI SRE Incident Response Environment.

    The agent observes a degraded production system and must diagnose
    and resolve the incident through a sequence of actions.

    Key design principles:
    - Hidden state: root cause is NOT revealed upfront
    - State evolves: delays worsen the system
    - Actions have consequences: some fixes cost, wrong ones penalize
    - Red herrings: not every alert is the real cause
    """

    def __init__(self, scenario: str = "easy", seed: int = 42):
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)

        # Hidden truth — agent must infer this
        self._hidden_root_cause: str = ""
        self._hidden_bad_service: str = ""
        self._hidden_bad_deploy_version: str = ""

        # Tracking
        self._step_count = 0
        self._max_steps = 20
        self._cumulative_reward = 0.0
        self._action_log: List[str] = []
        self._revealed_logs: List[str] = []
        self._metrics_checked: bool = False
        self._logs_checked: bool = False
        self._root_cause_inferred: bool = False
        self._correct_fix_applied: bool = False
        self._recovery_verified: bool = False
        self._done: bool = False
        self._time_elapsed: int = 0

        # Full hidden system state
        self._system: Dict[str, Any] = {}
        self._full_logs: List[str] = []

    def reset(self, task: Optional[str] = None) -> Observation:
    # If task is provided, override scenario
        if task:
         self.scenario = task

        self._step_count = 0
        self._cumulative_reward = 0.0
        self._action_log = []
        self._revealed_logs = []
        self._metrics_checked = False
        self._logs_checked = False
        self._root_cause_inferred = False
        self._correct_fix_applied = False
        self._recovery_verified = False
        self._done = False
        self._time_elapsed = 0

        self._setup_scenario()
        return self._build_observation()



    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._time_elapsed += self.rng.randint(10, 30)
        self._action_log.append(action.action_type)

        step_reward, breakdown, info = self._execute_action(action)

        # Time pressure: system degrades if agent is slow
        if self._step_count > 10 and not self._correct_fix_applied:
            self._degrade_system()

        done = self._check_done(action)
        self._done = done

        self._cumulative_reward += step_reward

        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            breakdown=breakdown,
            done=done,
            info=info,
        )
        return self._build_observation(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "step": self._step_count,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "action_log": self._action_log,
            "root_cause_inferred": self._root_cause_inferred,
            "correct_fix_applied": self._correct_fix_applied,
            "recovery_verified": self._recovery_verified,
            "system_snapshot": {
                k: v.dict() for k, v in self._system["services"].items()
            },
        }

    # ------------------------------------------------------------------ #
    #  Scenario Setup                                                      #
    # ------------------------------------------------------------------ #

    def _setup_scenario(self):
        if self.scenario == "easy":
            self._setup_easy()
        elif self.scenario == "medium":
            self._setup_medium()
        elif self.scenario == "hard":
            self._setup_hard()
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def _setup_easy(self):
        """Single overloaded service. Solution: scale_service."""
        self._hidden_root_cause = "cpu_spike"
        self._hidden_bad_service = "api"

        self._system = {
            "services": {
                "auth": ServiceState(cpu=35, memory=42, status=ServiceStatus.HEALTHY,
                                     error_rate=0.0, latency_ms=45),
                "api":  ServiceState(cpu=94, memory=71, status=ServiceStatus.OVERLOADED,
                                     error_rate=0.15, latency_ms=820),
                "db":   ServiceState(cpu=38, memory=55, status=ServiceStatus.HEALTHY,
                                     error_rate=0.0, latency_ms=22),
            },
            "deploy_versions": {"auth": "v3", "api": "v3", "db": "v3"},
            "alerts": ["[CRITICAL] api CPU at 94%", "[WARN] api latency > 800ms"],
        }

        self._full_logs = [
            "[api] Received 10x traffic spike from marketing campaign",
            "[api] Worker pool exhausted — requests queuing",
            "[auth] All systems nominal",                      # red herring
            "[db]  Query times normal",                        # red herring
            "[api] CPU throttling initiated",
        ]

    def _setup_medium(self):
        """Hidden OOM in db buried in noisy logs. Solution: check_logs → restart_pod(db)."""
        self._hidden_root_cause = "oom_crash"
        self._hidden_bad_service = "db"

        self._system = {
            "services": {
                "auth": ServiceState(cpu=51, memory=60, status=ServiceStatus.DEGRADED,
                                     error_rate=0.08, latency_ms=210),
                "api":  ServiceState(cpu=62, memory=58, status=ServiceStatus.DEGRADED,
                                     error_rate=0.12, latency_ms=430),
                "db":   ServiceState(cpu=44, memory=97, status=ServiceStatus.OVERLOADED,
                                     error_rate=0.35, latency_ms=2100),
            },
            "deploy_versions": {"auth": "v5", "api": "v5", "db": "v5"},
            "alerts": [
                "[WARN] auth response time elevated",
                "[WARN] api error rate 12%",
                "[WARN] db memory at 97%",
                "[INFO] Routine cron job completed",           # red herring
            ],
        }

        self._full_logs = [
            "[auth] Upstream timeout from db layer",
            "[api]  Retrying db connection (attempt 3/3)",
            "[db]   Connection pool: 0/50 available",
            "[cron] Backup job completed successfully",        # red herring
            "[db]   FATAL: Out of memory — killed process 4821",  # real cause
            "[api]  503 returned to clients",
            "[auth] Degraded mode: cached responses serving",
        ]

    def _setup_hard(self):
        """
        Cascading failure: bad deploy v6 broke auth → db overwhelmed → api down.
        Red herring: db memory looks high but auth is root.
        Solution: check_metrics → check_logs → rollback_deploy(auth) → verify_recovery → mark_resolved
        """
        self._hidden_root_cause = "bad_deploy"
        self._hidden_bad_service = "auth"
        self._hidden_bad_deploy_version = "v6"

        self._system = {
            "services": {
                "auth": ServiceState(cpu=88, memory=79, status=ServiceStatus.DEGRADED,
                                     error_rate=0.42, latency_ms=1800),
                "db":   ServiceState(cpu=91, memory=93, status=ServiceStatus.OVERLOADED,
                                     error_rate=0.28, latency_ms=3400),
                "api":  ServiceState(cpu=30, memory=41, status=ServiceStatus.DOWN,
                                     error_rate=1.0, latency_ms=0),
            },
            "deploy_versions": {"auth": "v6", "api": "v5", "db": "v5"},
            "alerts": [
                "[CRITICAL] api is DOWN",
                "[CRITICAL] db CPU at 91%",
                "[WARN] auth error rate 42%",
                "[WARN] auth deployed v6 at 14:32 UTC",
            ],
        }

        self._full_logs = [
            "[api]  Cannot reach auth service — refusing requests",
            "[db]   Connection storm: 8,000 reconnect attempts/sec",
            "[auth] v6 deploy introduced infinite retry loop on token refresh",  # root cause
            "[db]   Overwhelmed by auth retry storm",
            "[auth] Previous version v5 was stable",
            "[api]  Health check: FAIL (dependency: auth)",
        ]

    # ------------------------------------------------------------------ #
    #  Action Execution                                                    #
    # ------------------------------------------------------------------ #

    def _execute_action(self, action: Action) -> Tuple[float, Dict, Dict]:
        a = action.action_type
        svc = action.target_service
        reward = 0.0
        breakdown: Dict[str, float] = {}
        info: Dict[str, Any] = {"action": a, "target": svc}

        if a not in VALID_ACTIONS:
            breakdown["invalid_action"] = -0.3
            info["error"] = f"Unknown action: {a}"
            return -0.3, breakdown, info

        # --- Diagnostic actions ---
        if a == "check_metrics":
            self._metrics_checked = True
            reward += 0.05
            breakdown["metrics_checked"] = 0.05
            info["result"] = {
                k: {"cpu": v.cpu, "memory": v.memory, "error_rate": v.error_rate}
                for k, v in self._system["services"].items()
            }

        elif a == "check_logs":
            if not self._logs_checked:
                self._logs_checked = True
                self._revealed_logs = list(self._full_logs)
                reward += 0.1
                breakdown["logs_revealed"] = 0.1
                # If root cause log is revealed, mark inferred
                if self._hidden_root_cause in ["oom_crash", "bad_deploy"]:
                    self._root_cause_inferred = True
                    reward += 0.2
                    breakdown["root_cause_inferred"] = 0.2
                info["result"] = self._revealed_logs
            else:
                reward -= 0.05
                breakdown["repeated_action"] = -0.05
                info["result"] = self._revealed_logs

        # --- Fix actions ---
        elif a == "scale_service":
            if svc is None:
                return -0.1, {"no_target": -0.1}, {"error": "target_service required"}
            if self._hidden_root_cause == "cpu_spike" and svc == self._hidden_bad_service:
                self._correct_fix_applied = True
                self._system["services"][svc] = ServiceState(
                    cpu=38, memory=55, status=ServiceStatus.HEALTHY,
                    error_rate=0.01, latency_ms=60)
                reward += 0.4
                breakdown["correct_fix"] = 0.4
                info["result"] = f"{svc} scaled. CPU normalized."
            else:
                reward -= 0.1
                breakdown["unnecessary_scale"] = -0.1
                info["result"] = f"Scaling {svc} had no effect on incident."

        elif a == "restart_pod":
            if svc is None:
                return -0.1, {"no_target": -0.1}, {"error": "target_service required"}
            if self._hidden_root_cause == "oom_crash" and svc == self._hidden_bad_service:
                self._correct_fix_applied = True
                self._system["services"][svc] = ServiceState(
                    cpu=30, memory=48, status=ServiceStatus.HEALTHY,
                    error_rate=0.02, latency_ms=35)
                reward += 0.4
                breakdown["correct_fix"] = 0.4
                info["result"] = f"{svc} pod restarted. OOM cleared."
            elif svc == self._hidden_bad_service:
                reward += 0.1
                breakdown["partial_fix"] = 0.1
                info["result"] = f"Restart helped temporarily but not root cause."
            else:
                reward -= 0.15
                breakdown["wrong_service_restart"] = -0.15
                info["result"] = f"Restarting {svc} was not necessary."

        elif a == "rollback_deploy":
            if svc is None:
                svc = self._hidden_bad_service
            if self._hidden_root_cause == "bad_deploy" and svc == self._hidden_bad_service:
                self._correct_fix_applied = True
                self._system["services"]["auth"] = ServiceState(
                    cpu=30, memory=40, status=ServiceStatus.HEALTHY,
                    error_rate=0.0, latency_ms=45)
                self._system["services"]["db"] = ServiceState(
                    cpu=35, memory=52, status=ServiceStatus.HEALTHY,
                    error_rate=0.0, latency_ms=25)
                self._system["services"]["api"] = ServiceState(
                    cpu=28, memory=38, status=ServiceStatus.HEALTHY,
                    error_rate=0.0, latency_ms=55)
                self._system["deploy_versions"][svc] = "v5"
                reward += 0.4
                breakdown["correct_rollback"] = 0.4
                info["result"] = f"Rolled back {svc} to v5. Cascade resolving."
            else:
                reward -= 0.2
                breakdown["unnecessary_rollback"] = -0.2
                info["result"] = f"Rollback of {svc} caused brief downtime but didn't fix incident."

        elif a == "page_oncall":
            reward -= 0.1
            breakdown["paged_oncall"] = -0.1
            info["result"] = "On-call engineer paged. -0.1 penalty for escalation."

        elif a == "verify_recovery":
            if self._correct_fix_applied:
                self._recovery_verified = True
                reward += 0.15
                breakdown["recovery_verified"] = 0.15
                info["result"] = "All services healthy. Ready to mark resolved."
            else:
                reward -= 0.05
                breakdown["premature_verify"] = -0.05
                info["result"] = "System still degraded. Fix the root cause first."

        elif a == "mark_resolved":
            all_healthy = all(
                s.status == ServiceStatus.HEALTHY
                for s in self._system["services"].values()
            )
            if all_healthy and self._correct_fix_applied:
                reward += 0.2
                breakdown["correct_resolution"] = 0.2
                info["result"] = "Incident resolved correctly."
            elif not self._correct_fix_applied:
                reward -= 0.5
                breakdown["premature_resolution"] = -0.5
                info["result"] = "Marked resolved while system still broken!"
            else:
                reward -= 0.1
                breakdown["unverified_resolution"] = -0.1
                info["result"] = "Resolved but recovery not verified — risky."

        # Speed bonus: reward faster resolution
        if self._correct_fix_applied and self._step_count <= 6:
            speed_bonus = max(0.0, 0.15 - self._step_count * 0.02)
            reward += speed_bonus
            breakdown["speed_bonus"] = round(speed_bonus, 4)

        return round(reward, 4), breakdown, info

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _degrade_system(self):
        """Slowly worsen unhealthy services over time to add pressure."""
        for name, svc in self._system["services"].items():
            if svc.status != ServiceStatus.HEALTHY:
                svc.cpu = min(100, svc.cpu + self.rng.uniform(1, 4))
                svc.memory = min(100, svc.memory + self.rng.uniform(1, 3))
                svc.error_rate = min(1.0, svc.error_rate + 0.02)

    def _check_done(self, action: Action) -> bool:
        if self._step_count >= self._max_steps:
            return True
        if action.action_type == "mark_resolved":
            return True
        return False

    def _build_observation(self) -> Observation:
        return Observation(
            services=copy.deepcopy(self._system["services"]),
            alerts=list(self._system["alerts"]),
            visible_logs=list(self._revealed_logs),
            deploy_versions=dict(self._system["deploy_versions"]),
            step_number=self._step_count,
            incident_active=not self._correct_fix_applied,
            time_elapsed=self._time_elapsed,
            available_actions=VALID_ACTIONS,
        )
