# Scenario, simulator, judge settings
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field


class MedAdvisorTasksetConfig(vf.TasksetConfig):
    files: list[Path] = Field(min_length=1)
    system_message: str
    for_training: bool = True
    default_thinking: bool | None = None

class SimulatorConfig(vf.BaseConfig):
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"

    temperature: float = Field(0.0, ge=0)
    max_tokens: int = Field(2048, gt=0)
    timeout: float = Field(180.0, gt=0)
    max_retries: int = Field(2, ge=0)

class MedAdvisorJudgeConfig(vf.BaseConfig):
    model: str = "gpt-5.6-luna"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout: float = Field(180.0, gt=0)
    max_retries: int = Field(2, ge=0)
    concurrency: int = Field(64, gt=0)

class MedAdvisorRewardConfig(vf.BaseConfig):
    token_bonus_scale: float = Field(10.0, ge=0)
    token_bonus_cap: float = Field(0.015, ge=0)

class MedAdvisorEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "null"},
        runtime={"type": "subprocess"},
        network_notice=False,
    )
    simulator: SimulatorConfig = Field(
        default_factory=SimulatorConfig
    )
    judge: MedAdvisorJudgeConfig = Field(
        default_factory=MedAdvisorJudgeConfig
    )    
    reward: MedAdvisorRewardConfig = Field(
        default_factory=MedAdvisorRewardConfig
    )    