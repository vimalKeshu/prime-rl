from copy import deepcopy

import verifiers.v1 as vf
from openai import APIError
from pydantic import ValidationError

from med_advisor_rl.config import MedAdvisorEnvConfig
from med_advisor_rl.conversation import run_conversation
from med_advisor_rl.scenarios import MedAdvisorTask
from med_advisor_rl.simulator import (
    SimulatorSession,
    create_simulator_client,
)
from med_advisor_rl.rendering import configure_advisor_thinking
from med_advisor_rl.judge import JudgeOutputError, MedAdvisorJudge
from med_advisor_rl.reward import (
    NULL_REWARD,
    add_length_reward,
    compose_reward,
)

class MedAdvisorEnv(vf.Env[MedAdvisorEnvConfig]):
    _judge: MedAdvisorJudge | None = None

    async def start(self) -> None:
        self._judge = MedAdvisorJudge(self.config.judge)

    async def stop(self) -> None:
        if self._judge is not None:
            await self._judge.close()
            self._judge = None

    async def finalize(
        self,
        task: MedAdvisorTask,
        episode: vf.Episode,
    ) -> None:
        if len(episode.traces) != 1:
            raise ValueError(
                "MedAdvisorEnv expects exactly one advisor trace"
            )

        trace = episode.traces[0]

        if not trace.ok or trace.is_dropped:
            return

        conversation = trace.info["med_advisor"]
        chat = conversation["chat"]

        truncated = (
            conversation["end_reason"] == "advisor_truncated"
            or any(
                call.error is None and call.finish_reason == "length"
                for call in trace.calls
            )
        )

        if truncated:
            judgment = {
                "evaluation_status": "ROLLOUT_FAILURE",
                "evaluation_issue": "policy generation truncated",
            }
            reward = deepcopy(NULL_REWARD)
            reward.update(
                valid=True,
                trajectory_reward=-1.0,
                hard_failure=True,
                rollout_failure="generation_truncation",
            )

        elif not any(message["role"] == "assistant" for message in chat):
            issue = "trajectory has no assistant turns"
            judgment = {
                "evaluation_status": "ERROR",
                "evaluation_issue": issue,
            }
            reward = deepcopy(NULL_REWARD)
            reward["exclusion_reason"] = issue

        else:
            turns = []

            for node in trace.nodes:
                if not node.sampled:
                    continue

                if len(node.token_ids) == 0:
                    raise ValueError(
                        "Advisor trace is missing token IDs; "
                        "use a renderer-backed client for scoring"
                    )

                completion_ids = [
                    int(token)
                    for token, sampled in zip(
                        node.token_ids,
                        node.mask,
                        strict=True,
                    )
                    if sampled
                ]
                turns.append({"completion_ids": completion_ids})

            if not turns:
                raise ValueError(
                    "Advisor transcript has answers but no sampled nodes"
                )

            if self._judge is None:
                raise RuntimeError(
                    "Judge is not initialized; run the environment "
                    "through its serving lifecycle"
                )

            try:
                result = await self._judge.evaluate(task.data, chat)
            except (APIError, ValidationError, JudgeOutputError) as error:
                issue = f"{type(error).__name__}: {error}"
                judgment = {
                    "evaluation_status": "ERROR",
                    "evaluation_issue": issue,
                }
                reward = deepcopy(NULL_REWARD)
                reward["exclusion_reason"] = issue
            else:
                judgment = result.model_dump()
                reward = compose_reward(result)
                reward = add_length_reward(
                    reward,
                    {"turns": turns},
                    self.config.reward.model_dump(),
                )

        conversation["judge"] = judgment
        conversation["reward"] = reward

        if not reward["valid"]:
            trace.drop(reward["exclusion_reason"])
            return

        trace.record_reward(
            "med_advisor",
            reward["trajectory_reward"],
        )

        for name in ("quality_reward", "length_reward", "token_score"):
            value = reward.get(name)
            if value is not None:
                trace.record_metric(f"med_advisor/{name}", value)

    async def run(
        self,
        task: MedAdvisorTask,
        agents: vf.Agents,
    ) -> None:
        configure_advisor_thinking(
            agents.agent,
            thinking=task.data.thinking,
        )    
        async with create_simulator_client(
            self.config.simulator
        ) as client:
            simulator = SimulatorSession(
                client=client,
                config=self.config.simulator,
                data=task.data.simulator,
                opening=task.data.prompt,
            )

            async with agents.agent.interaction(task) as advisor:
                await run_conversation(
                    advisor=advisor,
                    simulator=simulator,
                    data=task.data,
                )