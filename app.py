"""
FastAPI server exposing the OpenEnv HTTP interface:
  POST /reset   → returns initial Observation
  POST /step    → accepts Action, returns (Observation, Reward, done, info)
  GET  /state   → returns current State
  GET  /health  → health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from env.environment import IncidentEnv
from env.models import Action

app = FastAPI(
    title="Incident Response Environment",
    description="OpenEnv-compliant RL environment for AI SRE training",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance
_env: Optional[IncidentEnv] = None


# ---------------- REQUEST MODELS ----------------

class StepRequest(BaseModel):
    action_type: str
    target_service: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}


# ---------------- ROUTES ----------------

@app.get("/")
def root():
    return {"message": "OpenEnv Incident Environment is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "env_initialized": _env is not None}


@app.post("/reset")
def reset(req: dict = {}) -> Dict[str, Any]:
    global _env

    task = req.get("task", "easy")

    task = req.get("task", "easy")

    _env = IncidentEnv(scenario=task)
    obs = _env.reset(task=task)

    return {
        "session_id": "default-session",
        "observation": obs.dict()
    }


@app.post("/step")
def step(req: StepRequest):
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    action = Action(
        action_type=req.action_type,
        target_service=req.target_service,
        parameters=req.parameters or {},
    )

    obs, reward, done, info = _env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    global _env

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    return _env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "CPU Spike Response",
                "description": "Traffic spike causing overload. Scale service.",
                "difficulty": "easy",
            },
            {
                "id": "medium",
                "name": "Hidden OOM Crash",
                "description": "Memory issue hidden in logs causing degradation.",
                "difficulty": "medium",
            },
            {
                "id": "hard",
                "name": "Cascading Failure",
                "description": "Multiple services failing after bad deploy.",
                "difficulty": "hard",
            },
        ]
    }