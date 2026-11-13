# Persona prompt and API client
import os

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from med_advisor_rl.config import SimulatorConfig
from med_advisor_rl.scenarios import SimulatorData


SIM_SYSTEM = """You are role-playing a person talking to a medical AI assistant.
You are the one asking. You are never the assistant.

WHO YOU ARE
  A {role} with {literacy} medical knowledge, currently {affect}.
  What you want: {goal}
  How you talk: {style}

WHAT YOU KNOW AND WILL SAY FREELY
{facts}

WHAT YOU KNOW BUT WILL NOT VOLUNTEER
  Say these only if the assistant asks about that specific thing. Do not
  offer them, and do not hint that you are holding anything back.
{withheld}

RULES
- Write only your next message: 1-3 sentences in your own voice. No narration,
  no labels, no stage directions.
- Never invent clinical facts beyond the lists above. If asked something not
  listed, say you do not know.
- Stay in character even if the assistant is wrong, evasive, or refuses you.
- Never mention that you are role-playing.
- When everything you came to ask has been answered, reply with exactly [[DONE]].
"""

def _bullets(items: dict[str, str]) -> str:
    return "\n".join(
        f"  - {key}: {value}" for key, value in items.items()
    ) or "  - (nothing)"

def build_simulator_prompt(data: SimulatorData) -> str:
    return SIM_SYSTEM.format(
        **data.persona,
        facts=_bullets(data.facts),
        withheld=_bullets(data.withheld),
    )

def create_simulator_client(config: SimulatorConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=os.environ[config.api_key_env],
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


class SimulatorSession:
    def __init__(
        self,
        client: AsyncOpenAI,
        config: SimulatorConfig,
        data: SimulatorData,
        opening: str,
    ) -> None:
        self.client = client
        self.config = config
        self.messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": build_simulator_prompt(data),
            },
            {
                "role": "assistant",
                "content": opening,
            },
        ]

    async def reply(self, advisor_answer: str) -> str:
        self.messages.append(
            {"role": "user", "content": advisor_answer}
        )

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        text = (response.choices[0].message.content or "").strip()
        self.messages.append(
            {"role": "assistant", "content": text}
        )
        return text
