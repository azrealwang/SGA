from __future__ import annotations

from vlm.eval.answer_normalization import normalize_answer


def exact_match_correct(prediction: str, ground_truth_answers: list[str]) -> bool:
    p = normalize_answer(prediction)
    gts = {normalize_answer(a) for a in ground_truth_answers}
    return p in gts


def vqav2_soft_score(prediction: str, ground_truth_answers: list[str]) -> float:
    """
    VQAv2 accuracy: min(#humans that said this / 3, 1) after normalization.
    Here `ground_truth_answers` is the list of human answers for the question.
    """
    pred = normalize_answer(prediction)
    matches = sum(1 for a in ground_truth_answers if normalize_answer(a) == pred)
    return min(matches / 3.0, 1.0)


def asr_conditional(
    clean_correct: list[bool],
    adv_wrong: list[bool],
) -> float:
    """
    ASR = (# clean correct and adv wrong) / (# clean correct).
    Lists are aligned per sample on the attack set.
    """
    num_cc = sum(1 for c in clean_correct if c)
    if num_cc == 0:
        return float("nan")
    succ = sum(1 for c, w in zip(clean_correct, adv_wrong) if c and w)
    return succ / num_cc
