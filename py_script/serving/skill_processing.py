import re

ALLOWED_SKILLS = {"PLACE_ON", "PLACE_IN", "PICKUP_FROM", "OPEN", "CLOSE", "TURN_ON", "TURN_OFF"}
SKILL_EXPR_RE = re.compile(r"^([A-Z_]+)\((.*)\)$")
def get_skill_name(skill_text: str) -> str:
    """Strip the (...) parameters off a skill expression, leaving the bare verb."""
    skill_text = (skill_text or "").strip()
    match = SKILL_EXPR_RE.match(skill_text)
    if not match:
        return skill_text.split("(", 1)[0].strip().upper()
    return match.group(1).strip().upper()


PLAN_ITEM_RE = re.compile(r"\d+\.\s*(?=[A-Z_]+\()")
def parse_plan(plan_str: str) -> list[str]:
    """Parse a numbered plan string like ``"1. PICKUP_FROM(...) 2. PLACE_ON(...)"`` into
    a list of skill expressions. Mirrors the format the annotation pipeline emits in
    ``skill_annotations.json`` so the trained model sees the same skill strings it
    saw during training.
    """
    cleaned = (plan_str or "").strip()
    if not cleaned:
        return []
    # Split on the "<num>. " markers; the first piece is anything before "1." (likely empty).
    parts = PLAN_ITEM_RE.split(cleaned)
    skills: list[str] = []
    for part in parts:
        item = part.strip().rstrip(",;")
        if not item:
            continue
        match = SKILL_EXPR_RE.match(item)
        if not match:
            continue
        name = match.group(1).strip().upper()
        if name not in ALLOWED_SKILLS:
            continue
        # Re-emit canonical form (keeps spacing inside the parens as Gemini wrote it).
        skills.append(item)
    return skills

