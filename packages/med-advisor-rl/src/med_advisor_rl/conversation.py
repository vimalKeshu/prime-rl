# Advisor/simulator turn loop
import verifiers.v1 as vf

from med_advisor_rl.scenarios import MedAdvisorData
from med_advisor_rl.simulator import SimulatorSession


async def run_conversation(
    advisor: vf.Interaction,
    simulator: SimulatorSession,
    data: MedAdvisorData,
) -> None:
    chat = [{"role": "user", "content": data.prompt}]
    result = {
        "chat": chat,
        "end_reason": "max_turns",
    }
    advisor.trace.info["med_advisor"] = result

    user_message: str | None = None

    for turn_index in range(data.max_turns):
        calls_before = len(advisor.trace.calls)
        segment = await advisor.turn(user_message)

        if advisor.trace.errors:
            result["end_reason"] = "advisor_error"
            return

        new_calls = advisor.trace.calls[calls_before:]
        if any(
            call.error is None and call.finish_reason == "length"
            for call in new_calls
        ):
            result["end_reason"] = "advisor_truncated"
            return

        if segment.terminated:
            raise RuntimeError(
                "Advisor interaction ended before producing a response"
            )

        answer = segment.last_reply
        if not answer:
            result["end_reason"] = "advisor_no_answer"
            return

        chat.append({"role": "assistant", "content": answer})

        if turn_index == data.max_turns - 1:
            return

        user_message = await simulator.reply(answer)

        if not user_message:
            result["end_reason"] = "persona_no_response"
            return

        if "[[DONE]]" in user_message:
            result["end_reason"] = "persona_satisfied"
            return

        chat.append({"role": "user", "content": user_message})