# Visible-history/token handling
import re

from renderers import DefaultRendererConfig, ParsedResponse, ToolSpec
from renderers.default import DefaultRenderer
from dataclasses import replace

import verifiers.v1 as vf


_THINK = re.compile(r"<think>(.*?)</think>", re.S)

def configure_advisor_thinking(
    agent: vf.Agent,
    *,
    thinking: bool,
) -> None:
    params = agent.ctx.sampling.model_dump(exclude_none=True)
    template_kwargs = dict(params.get("chat_template_kwargs") or {})
    template_kwargs["enable_thinking"] = thinking
    params["chat_template_kwargs"] = template_kwargs

    sampling = vf.SamplingConfig.model_validate(params)

    agent.config = agent.config.model_copy(
        update={"sampling": sampling}
    )
    agent.ctx = replace(agent.ctx, sampling=sampling)

class MedAdvisorRenderer(DefaultRenderer):
    config_class = DefaultRendererConfig

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        if tools:
            raise ValueError("MedAdvisorRenderer does not support tools")

        text = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

        if "<think>" in text and "</think>" not in text:
            return ParsedResponse(
                content="",
                reasoning_content=text.split("<think>", 1)[1].strip(),
                reasoning_complete=False,
            )

        reasoning = "\n".join(
            block.strip() for block in _THINK.findall(text)
        )
        answer = _THINK.sub("", text).strip()

        return ParsedResponse(
            content=answer,
            reasoning_content=reasoning or None,
            reasoning_complete=True,
        )
