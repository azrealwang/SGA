"""Unified VQA question wording across backends."""

SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def format_question(question: str, prompt_mode: str = "short_answer") -> str:
    q = question.strip()
    if prompt_mode == "short_answer":
        return q + SHORT_ANSWER_SUFFIX
    if prompt_mode == "raw":
        return q
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
