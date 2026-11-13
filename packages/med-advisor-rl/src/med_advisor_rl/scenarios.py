import json
from typing import Any

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field

from med_advisor_rl.config import MedAdvisorTasksetConfig


class SimulatorData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    persona: dict[str, str]
    facts: dict[str, str]
    withheld: dict[str, str] = Field(default_factory=dict)


class MedAdvisorData(vf.TaskData):
    id: str
    name: str
    prompt: str
    system_prompt: str

    domain: str
    max_turns: int = Field(gt=0)
    thinking: bool
    simulator: SimulatorData

    oracle: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

def card_to_data(
    card: dict[str, Any],
    *,
    system_message: str,
    for_training: bool,
    default_thinking: bool | None = None,
) -> MedAdvisorData:
    scenario_id = card.get("id", "<unknown>")
    provenance = card.get("provenance") or {}

    if for_training:
        forbidden = (
            provenance.get("dataset_split") != "train"
            or provenance.get("evaluation_only", False)
            or provenance.get("training_use_forbidden", False)
        )
        if forbidden:
            raise ValueError(
                f"Scenario {scenario_id!r} is not permitted for training"
            )

    thinking = card.get("thinking", default_thinking)
    if thinking is None:
        raise ValueError(
            f"Scenario {scenario_id!r} has no thinking mode; "
            "set it on the card or provide default_thinking"
        )

    simulator = dict(card["simulator"])
    opening = simulator.pop("opening")
    simulator["withheld"] = simulator.get("withheld") or {}

    return MedAdvisorData(
        id=card["id"],
        name=card["title"],
        prompt=opening,
        system_prompt=system_message,
        domain=card["domain"],
        max_turns=card["max_turns"],
        thinking=thinking,
        simulator=SimulatorData.model_validate(simulator),
        oracle=card.get("oracle") or {},
        provenance=provenance,
    )    
    
class MedAdvisorTask(vf.Task[MedAdvisorData]):
    pass

class MedAdvisorTaskset(
    vf.Taskset[MedAdvisorTask, MedAdvisorTasksetConfig]
):
    def load(self) -> list[MedAdvisorTask]:
        tasks: list[MedAdvisorTask] = []
        seen_ids: set[str] = set()

        for path in self.config.files:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue

                    data = card_to_data(
                        json.loads(line),
                        system_message=self.config.system_message,
                        for_training=self.config.for_training,
                        default_thinking=self.config.default_thinking,
                    )

                    if data.id in seen_ids:
                        raise ValueError(
                            f"Duplicate scenario ID {data.id!r} "
                            f"at {path}:{line_number}"
                        )
                    seen_ids.add(data.id)

                    tasks.append(
                        MedAdvisorTask(data, self.config.task)
                    )

        if not tasks:
            raise ValueError("No scenario cards found in configured files")

        return tasks
