# Pure reward calculation
from typing import Any

from med_advisor_rl.judge import JudgeResult
from med_advisor_rl.rubrics import RUBRIC

NULL_REWARD = {
    "valid": False,
    "exclusion_reason": "",
    "quality_reward": None,
    "trajectory_reward": None,
    "hard_failure": False,
    "hard_failure_dimensions": [],
    "dimension_scores": {},
    "active_weight": 0.0,
    "reward_version": "v7",
    "reward_terms": {},
    "length_reward": None,
    "token_score": None,
    "raw_length_score": None,
    "completion_tokens": 0,
    "completion_turns": 0,
}

QUALITY_UTILITY = {
    "A": 1.00,
    "B": 0.85,
    "C": 0.60,
    "D": 0.25,
    "E": 0.00,
}

CATEGORY_WEIGHTS = {
    "correctness": 0.50,
    "conversation_execution": 0.40,
    "presentation": 0.10,
}

CATEGORY_DIMENSIONS = {
    "correctness": {
        "medical_accuracy": 1.0,
        "medication_and_numerical_integrity": 1.0,
        "evidence_integrity": 1.0,
        "safety_boundary": 1.0,         
    },
    "conversation_execution": {
        "uncertainty_calibration": 1.0,
        "user_question_fulfillment": 1.0,
        "appropriate_completeness": 1.0,        
        "audience_adaptation": 1.0,
        "follow_up_quality": 1.0,
        "multi_turn_coherence": 1.0,
    },    
    "presentation": {
        "answer_format": 1.0,
    },
}

REQUIRED_PRIMARY_DIMENSIONS = {
    "medical_accuracy",

    "uncertainty_calibration",
    "user_question_fulfillment",
    "appropriate_completeness",
    "audience_adaptation",
    "follow_up_quality",

    "answer_format",
}

DIMENSION_WEIGHTS = {
    name: CATEGORY_WEIGHTS[category] * weight / sum(weights.values())
    for category, weights in CATEGORY_DIMENSIONS.items()
    for name, weight in weights.items()
}

assert RUBRIC.keys() == DIMENSION_WEIGHTS.keys(), (
    f"Rubric/scorer mismatch: {RUBRIC.keys() ^ DIMENSION_WEIGHTS.keys()}"
)

def compose_reward(result: JudgeResult) -> dict[str, Any]:
    reward = {
        "valid": False,
        "exclusion_reason": "",
        "quality_reward": None,       # Base quality BEFORE the gate/override.
        "trajectory_reward": None,    # The value the trainer must use.
        "hard_failure": False,
        "hard_failure_dimensions": [],
        "dimension_scores": {},
        "active_weight": 0.0,         # Diagnostic only; not a denominator below.
        "reward_version": "v7",
        "category_scores": {},
        "essential_quality": None,
        "limiting_dimensions": [],
        "reward_terms": {},
        "length_reward": None,
        "token_score": None,
        "raw_length_score": None,
        "completion_tokens": 0,
        "completion_turns": 0,
    }

    if result.evaluation_status != "VALID":
        reward["exclusion_reason"] = (
            result.evaluation_issue or "insufficient judge input"
        )
        return reward

    grades = result.grades.model_dump()

    if grades.keys() != DIMENSION_WEIGHTS.keys():
        reward["exclusion_reason"] = (
            "judge dimensions do not match scorer dimensions"
        )
        return reward

    scores = {
        name: None if item["grade"] == "NA" else QUALITY_UTILITY[item["grade"]]
        for name, item in grades.items()
    }

    reward["dimension_scores"] = scores
    reward["active_weight"] = sum(
        DIMENSION_WEIGHTS[name]
        for name, score in scores.items()
        if score is not None
    )

    for name in REQUIRED_PRIMARY_DIMENSIONS:
        if scores[name] is None:
            reward["exclusion_reason"] = (
                f"required primary grade is NA: {name}"
            )
            return reward

    category_scores = {}

    for category, weights in CATEGORY_DIMENSIONS.items():
        active = [
            (scores[name], weight)
            for name, weight in weights.items()
            if scores[name] is not None
        ]

        if not active:
            reward["exclusion_reason"] = (
                f"no applicable grades in {category}"
            )
            return reward

        category_scores[category] = (
            sum(score * weight for score, weight in active)
            / sum(weight for _, weight in active)
        )

    # ("correctness", "conversation_execution", "presentation")
    q1 = category_scores["correctness"]
    q2 = category_scores["conversation_execution"]
    q3 = category_scores["presentation"]
    reward_terms = {
        "correctness": CATEGORY_WEIGHTS["correctness"] * q1,
        "conversation_execution": (
            CATEGORY_WEIGHTS["conversation_execution"] * q1 * q2
        ),
        "presentation": CATEGORY_WEIGHTS["presentation"] * q1 * q2 * q3,
    }
    cascade_reward = sum(reward_terms.values())

    reward.update(
        valid=True,
        quality_reward=cascade_reward,
        trajectory_reward=cascade_reward,
        category_scores=category_scores,
        reward_terms=reward_terms,
    )

    return reward                            


def add_length_reward(reward: dict[str, Any], trajectory: dict, cfg: dict) -> dict[str, Any]:
    """Scale a bounded token score by judged quality before adding it.

    1 / (average tokens per turn * number of turns) is exactly 1 / total
    completion tokens. Counting completion_ids includes reasoning tokens.
    """
    if not reward["valid"] or reward["quality_reward"] is None:
        return reward

    token_counts = [len(turn.get("completion_ids") or [])
                    for turn in trajectory.get("turns", [])]
    total_tokens = sum(token_counts)
    if not total_tokens:
        return reward

    scale = float(cfg["token_bonus_scale"])
    cap = float(cfg["token_bonus_cap"])
    if scale < 0 or cap < 0:
        raise ValueError("token bonus scale and cap must be nonnegative")

    raw_score = 1.0 / total_tokens
    token_score = min(cap, scale * raw_score)
    bonus = reward["quality_reward"] * token_score
    reward["raw_length_score"] = raw_score
    reward["token_score"] = token_score
    reward["length_reward"] = bonus
    reward["completion_tokens"] = total_tokens
    reward["completion_turns"] = len(token_counts)
    reward["reward_terms"]["token_bonus"] = bonus
    reward["trajectory_reward"] = reward["quality_reward"] + bonus
    return reward
