"""
inference.py — Baseline LLM agent for incident-response-env

Uses OpenAI client (as required by OpenEnv spec).
Reads credentials from environment variables:
  - OPENAI_API_KEY or HF_TOKEN
  - API_BASE_URL  (default: https://api.openai.com/v1)
  - MODEL_NAME    (default: gpt-4o-mini)

Usage:
  python inference.py
  python inference.py --scenario easy
  python inference.py --all
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Config from env vars ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", HF_TOKEN)
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=OPENAI_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

You will receive:
- Current state of all services (CPU, memory, status, error rate)
- Active alerts
- Visible logs (may be empty until you check_logs)
- Deploy versions per service
- Your action history

Your job: diagnose the root cause and resolve the incident with minimum actions.

Available actions:
- check_logs           : reveal system logs (do this early)
- check_metrics        : get detailed metrics snapshot
- scale_service        : scale up a service (use: target_service=<name>)
- restart_pod          : restart a service pod (use: target_service=<name>)
- rollback_deploy      : rollback deploy to previous version (use: target_service=<name>)
- page_oncall          : page on-call engineer (penalty, use as last resort)
- verify_recovery      : verify all services recovered before closing
- mark_resolved        : close the incident (only when all services healthy)

Rules:
1. Always check_logs and check_metrics before acting
2. Target the ROOT CAUSE service, not symptoms
3. Only mark_resolved when all services are healthy
4. Avoid unnecessary actions — each wrong action costs points

Respond with ONLY a JSON object:
{
  "reasoning": "<brief thought>",
  "action_type": "<action>",
  "target_service": "<service or null>"
}
"""


def call_env(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=data, timeout=30)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[ERROR] Env call failed: {e}")
        sys.exit(1)


def format_observation(obs: Dict, action_history: List[str]) -> str:
    services = obs.get("services", {})
    service_lines = []
    for name, svc in services.items():
        service_lines.append(
            f"  {name}: cpu={svc['cpu']:.0f}% mem={svc['memory']:.0f}% "
            f"status={svc['status']} error_rate={svc['error_rate']:.0%} latency={svc['latency_ms']:.0f}ms"
        )

    logs = obs.get("visible_logs", [])
    log_section = "\n".join(f"  {l}" for l in logs) if logs else "  (not yet retrieved — use check_logs)"

    alerts = obs.get("alerts", [])
    alert_section = "\n".join(f"  {a}" for a in alerts)

    deploys = obs.get("deploy_versions", {})
    deploy_section = ", ".join(f"{k}={v}" for k, v in deploys.items())

    history_section = ", ".join(action_history[-6:]) if action_history else "none"

    return f"""
=== INCIDENT STATE (Step {obs.get('step_number', 0)}) ===
Services:
{chr(10).join(service_lines)}

Alerts:
{alert_section}

Deploy versions: {deploy_section}

Logs:
{log_section}

Recent actions: {history_section}
Time elapsed: {obs.get('time_elapsed', 0)}s
"""


def get_llm_action(observation_text: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": observation_text},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[WARN] LLM parse error: {e}. Using fallback.")
        return {"action_type": "check_logs", "target_service": None, "reasoning": "fallback"}


def run_scenario(scenario: str) -> float:
    print(f"\n{'='*60}")
    print(f"  Running scenario: {scenario.upper()}")
    print(f"{'='*60}")

    obs = call_env("/reset", "POST", {"scenario": scenario, "seed": 42})
    action_history: List[str] = []
    step = 0
    done = False

    while not done and step < 20:
        step += 1
        obs_text = format_observation(obs, action_history)
        llm_response = get_llm_action(obs_text)

        action_type    = llm_response.get("action_type", "check_logs")
        target_service = llm_response.get("target_service")
        reasoning      = llm_response.get("reasoning", "")

        print(f"\n[Step {step}] Reasoning: {reasoning}")
        print(f"         Action: {action_type}" + (f"({target_service})" if target_service else ""))

        result = call_env("/step", "POST", {
            "action_type": action_type,
            "target_service": target_service,
            "parameters": {},
        })

        obs   = result["observation"]
        rew   = result["reward"]
        done  = result["done"]
        info  = result["info"]

        action_history.append(action_type)
        print(f"         Reward: {rew['step_reward']:+.3f} | Cumulative: {rew['cumulative_reward']:+.3f}")
        if info.get("result"):
            print(f"         Result: {info['result']}")

        time.sleep(0.2)  # rate limiting

    # Final grading
    final_state = call_env("/state")
    from graders.graders import grade_easy, grade_medium, grade_hard

    action_log = final_state["action_log"]
    if scenario == "easy":
        score = grade_easy(action_log, final_state)
    elif scenario == "medium":
        score = grade_medium(action_log, final_state)
    else:
        score = grade_hard(action_log, final_state)

    print(f"\n{'─'*40}")
    print(f"  FINAL SCORE ({scenario}): {score:.4f} / 1.0")
    print(f"  Cumulative reward:       {final_state['cumulative_reward']:+.4f}")
    print(f"  Actions taken:           {action_log}")
    print(f"{'─'*40}")
    return score


def main():
    parser = argparse.ArgumentParser(description="Incident Response Env — Baseline Inference")
    parser.add_argument("--scenario", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--all", action="store_true", help="Run all 3 scenarios")
    args = parser.parse_args()

    scenarios = ["easy", "medium", "hard"] if args.all or args.scenario is None else [args.scenario]

    scores = {}
    for s in scenarios:
        scores[s] = run_scenario(s)

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for s, sc in scores.items():
        bar = "█" * int(sc * 20)
        print(f"  {s:<8} {sc:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'avg':<8} {avg:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
