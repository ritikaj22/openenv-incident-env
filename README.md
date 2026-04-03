# 🚨 incident-response-env

> An OpenEnv-compliant RL environment where an AI SRE agent diagnoses and resolves production incidents in a stateful simulated system.

---

## Motivation

Production incidents are high-stakes, time-critical, and require multi-step reasoning under uncertainty. Unlike static benchmarks, this environment forces an agent to:

- **Explore before acting** — logs are hidden by default
- **Reason causally** — symptoms ≠ root cause
- **Handle red herrings** — misleading alerts and logs
- **Act under time pressure** — system degrades with delay
- **Verify before closing** — premature resolution is penalized

This fills a real gap in RL agent evaluation: there are no good open environments for SRE/ops reasoning.

---

## Environment Description

The agent operates a simulated production system with 3 microservices: `auth`, `db`, and `api`. An incident is active and the agent must diagnose the root cause and restore all services to healthy status.

### Key design principles

- **Hidden state**: Root cause is not revealed upfront. Agent must `check_logs` to discover it.
- **State evolution**: Delayed action causes system degradation (CPU/memory creep).
- **Causal chain**: In hard mode, `auth → db → api` cascade — fixing a symptom doesn't resolve the incident.
- **Red herrings**: Noisy alerts and misleading logs force genuine reasoning.
- **Consequences**: Wrong rollbacks cause brief downtime; paging oncall incurs penalty.

---

## Action Space

| Action | Description | Requires `target_service` |
|--------|-------------|--------------------------|
| `check_logs` | Reveal system log lines (hidden by default) | No |
| `check_metrics` | Get detailed metrics snapshot | No |
| `scale_service` | Scale up a service instance | Yes |
| `restart_pod` | Restart a service pod | Yes |
| `rollback_deploy` | Rollback to previous deploy version | Yes |
| `page_oncall` | Escalate to on-call engineer (penalty) | No |
| `verify_recovery` | Confirm all services healthy | No |
| `mark_resolved` | Close incident (ends episode) | No |

---

## Observation Space

```json
{
  "services": {
    "auth": {"cpu": 35, "memory": 42, "status": "healthy", "error_rate": 0.0, "latency_ms": 45},
    "db":   {"cpu": 91, "memory": 93, "status": "overloaded", "error_rate": 0.28, "latency_ms": 3400},
    "api":  {"cpu": 30, "memory": 41, "status": "down", "error_rate": 1.0, "latency_ms": 0}
  },
  "alerts": ["[CRITICAL] api is DOWN", "[WARN] auth deployed v6 at 14:32 UTC"],
  "visible_logs": [],
  "deploy_versions": {"auth": "v6", "api": "v5", "db": "v5"},
  "step_number": 1,
  "incident_active": true,
  "time_elapsed": 42,
  "available_actions": ["check_logs", "check_metrics", ...]
}
```

---

## Tasks

### 🟢 Easy — CPU Spike Response
- **Scenario**: Marketing campaign causes 10x traffic spike on `api`. CPU at 94%.
- **Root cause**: CPU overload
- **Optimal solution**: `check_metrics` → `scale_service(api)` → `mark_resolved`
- **Expected score**: 0.9–1.0 for optimal path

### 🟡 Medium — Hidden OOM Crash
- **Scenario**: `db` crashed with OOM. Noisy logs hide the real cause. Upstream `auth` and `api` degraded.
- **Root cause**: Out-of-memory in `db`
- **Optimal solution**: `check_logs` → `restart_pod(db)` → `verify_recovery` → `mark_resolved`
- **Expected score**: 0.85–1.0 for optimal path

### 🔴 Hard — Cascading Deploy Failure
- **Scenario**: Bad deploy `v6` to `auth` introduces infinite retry loop → overwhelms `db` → `api` goes down.
- **Root cause**: Bad deploy on `auth`
- **Optimal solution**: `check_metrics` → `check_logs` → `rollback_deploy(auth)` → `verify_recovery` → `mark_resolved`
- **Expected score**: 0.9–1.0 for full correct path; 0.3–0.5 for partial (e.g. restart instead of rollback)

---

## Reward Function

```
reward = 0

# Diagnostics
+0.05  check_metrics
+0.10  check_logs (first time)
+0.20  root cause correctly inferred from logs

# Correct fix
+0.40  correct action on correct service

# Resolution
+0.15  verify_recovery before closing
+0.20  mark_resolved when all healthy
+0.15  speed bonus (max, decays with steps)

# Penalties
-0.05  repeated check_logs
-0.10  page_oncall
-0.15  restart wrong service
-0.20  unnecessary rollback
-0.50  mark_resolved while system still broken
```

---

## Setup & Usage

### Local development

```bash
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload --port 7860

# In another terminal, run baseline inference
python inference.py --all
```

### Docker

```bash
docker build -t incident-response-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  incident-response-env
```

### API usage

```bash
# Reset to easy scenario
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario": "easy", "seed": 42}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "check_logs"}'

# Get current state
curl http://localhost:7860/state
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `HF_TOKEN` | HuggingFace token (fallback) | — |
| `API_BASE_URL` | LLM API base URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `ENV_URL` | Environment server URL (for inference.py) | `http://localhost:7860` |

---

## Baseline Scores

| Scenario | Score | Notes |
|----------|-------|-------|
| easy     | ~0.85 | gpt-4o-mini, seed=42 |
| medium   | ~0.70 | gpt-4o-mini, seed=42 |
| hard     | ~0.55 | gpt-4o-mini, seed=42 |

---

## Project Structure

```
incident-response-env/
├── app.py                 # FastAPI server
├── inference.py           # Baseline LLM agent (OpenAI client)
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py          # Pydantic models (Observation, Action, Reward)
│   └── environment.py     # Core IncidentEnv logic
└── graders/
    ├── __init__.py
    └── graders.py         # Deterministic graders for all 3 tasks
```
