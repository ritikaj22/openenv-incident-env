from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    DOWN = "down"


class ServiceState(BaseModel):
    cpu: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    status: ServiceStatus
    error_rate: float = Field(default=0.0, ge=0, le=1, description="Request error rate 0-1")
    latency_ms: float = Field(default=50.0, description="Average response latency in ms")


class Observation(BaseModel):
    services: Dict[str, ServiceState]
    alerts: List[str]
    visible_logs: List[str]
    deploy_versions: Dict[str, str]
    step_number: int
    incident_active: bool
    time_elapsed: int = Field(description="Seconds since incident started")
    available_actions: List[str]


class Action(BaseModel):
    action_type: str = Field(..., description="One of the valid action types")
    target_service: Optional[str] = Field(None, description="Service to act on")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    breakdown: Dict[str, float]
    done: bool
    info: Dict[str, Any]
