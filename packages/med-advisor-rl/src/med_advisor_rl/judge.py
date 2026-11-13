from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, create_model

from med_advisor_rl.config import MedAdvisorJudgeConfig
from med_advisor_rl.rubrics import JUDGE_PROMPT, RUBRIC

if TYPE_CHECKING:
    from med_advisor_rl.scenarios import MedAdvisorData


Grade = Literal["A", "B", "C", "D", "E", "NA"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DimensionGrade(StrictModel):
    grade: Grade
    reason: str


Grades = create_model(
    "Grades",
    __base__=StrictModel,
    **{
        dimension: (DimensionGrade, ...)
        for dimension in RUBRIC
    },
)


class JudgeResult(StrictModel):
    evaluation_status: Literal["VALID", "INSUFFICIENT_INPUT"]
    evaluation_issue: str
    grades: Grades


def serialize(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def build_judge_input(
    data: MedAdvisorData,
    chat: list[dict[str, str]],
) -> str:
    scenario = {
        "title": data.name,
        "domain": data.domain,
        "audience": data.simulator.persona,
        "user_facts_volunteered": data.simulator.facts,
    }

    return (
        f"<SCENARIO>\n{serialize(scenario)}\n</SCENARIO>\n\n"
        f"<TRAJECTORY>\n{serialize(chat)}\n</TRAJECTORY>"
    )

class JudgeOutputError(RuntimeError):
    pass

class MedAdvisorJudge:
    def __init__(self, config: MedAdvisorJudgeConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            api_key=os.environ[config.api_key_env],
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self.gate = asyncio.Semaphore(config.concurrency)

    async def evaluate(
        self,
        data: MedAdvisorData,
        chat: list[dict[str, str]],
    ) -> JudgeResult:
        judge_input = build_judge_input(data, chat)

        async with self.gate:
            response = await self.client.responses.parse(
                model=self.config.model,
                instructions=JUDGE_PROMPT,
                input=judge_input,
                text_format=JudgeResult,
                store=False,
            )

        if response.output_parsed is None:
            raise JudgeOutputError(
                response.output_text
                or "judge returned no parsed result"
            )

        return response.output_parsed

    async def close(self) -> None:
        await self.client.close()