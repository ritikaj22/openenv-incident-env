🚀 OpenEnv Incident Response Environment

A production-grade simulation of real-world system failures, designed to evaluate intelligent agents in diagnosing and resolving infrastructure incidents.

🌌 Overview

Modern systems fail in complex, unpredictable ways.

This environment simulates:

⚠️ Traffic spikes
💥 Memory crashes
🔗 Cascading failures across services

Agents must:

observe → reason → act → recover → resolve

🧠 What Makes This Real
Hidden root causes (not directly visible)
Noisy logs with red herrings
Delayed system degradation
Multi-step recovery process
Action consequences (good & bad)
🏗️ System Architecture
Agent (inference.py)
        ↓
FastAPI (app.py)
        ↓
IncidentEnv (environment.py)
        ↓
State + Reward + Evolution
⚙️ API Endpoints
Endpoint	Description
/reset	Start new incident
/step	Take action
/state	Current system state
/health	Health check
🎯 Tasks
Task	Description	Difficulty
Easy	CPU spike in API	⭐
Medium	Hidden OOM in DB	⭐⭐
Hard	Cascading deploy failure	⭐⭐⭐
🧪 Action Space
check_logs
check_metrics
scale_service
restart_pod
rollback_deploy
verify_recovery
mark_resolved
📊 Observation Space
Service metrics (CPU, memory, latency)
Alerts
Logs (partial → full)
Deployment versions
🎯 Reward Design
for correct diagnosis
for correct fix
for fast resolution
− for wrong/redundant actions
− for premature resolution
🤖 Baseline Agent

A deterministic rule-based agent:

Reads logs
Detects patterns
Applies fixes
Produces reproducible scores
📈 Example Output
[START]
task: easy

[STEP]
step: 1
action: check_logs
reward: 0.1

...

[END]
score: 1.0
🐳 Running Locally
pip install -r requirements.txt
uvicorn app:app --port 7860
🧪 Run Inference
python inference.py
🌐 Live Deployment

👉 https://ritikaj22-openenv-incident-env.hf.space

💡 Why This Matters

This environment can be used to:

Evaluate AI agents in real-world scenarios
Benchmark decision-making under uncertainty
Simulate production incidents safely