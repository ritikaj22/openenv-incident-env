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
from typing import Any, Dict, Optional, Tuple

from env.environment import IncidentEnv
from env.models import Action, Observation, Reward

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

# Global env instance (single-session for simplicity)
_env: Optional[IncidentEnv] = None


class ResetRequest(BaseModel):
    scenario: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action_type: str
    target_service: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
@app.get("/")
def root():
    return {"message": "OpenEnv Incident Environment is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "env_initialized": _env is not None}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    global _env
    _env = IncidentEnv(scenario=req.scenario, seed=req.seed)
    obs = _env.reset()
    return obs.dict()


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    action = Action(
        action_type=req.action_type,
        target_service=req.target_service,
        parameters=req.parameters or {},
    )

    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs.dict(),
        reward=reward.dict(),
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "CPU Spike Response",
                "description": "Single service overloaded by traffic spike. Diagnose and scale.",
                "difficulty": "easy",
                "max_steps": 20,
            },
            {
                "id": "medium",
                "name": "Hidden OOM Crash",
                "description": "Out-of-memory crash buried in noisy logs causing upstream degradation.",
                "difficulty": "medium",
                "max_steps": 20,
            },
            {
                "id": "hard",
                "name": "Cascading Deploy Failure",
                "description": "Bad deploy triggers cascading failure across 3 services. Full diagnostic + rollback required.",
                "difficulty": "hard",
                "max_steps": 20,
            },
        ]
    }
