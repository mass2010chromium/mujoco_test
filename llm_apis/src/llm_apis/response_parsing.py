import json
import re

def extract_in_backticks(text: str, tag: str) -> str:
    """Look for a triple backticks block with a given tag, and return it."""
    code_block = re.search(f"```(?:{tag})?\\s*(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    return None


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from a VLM response that may contain markdown fences."""
    # Strip markdown code fences if present
    code_block = extract_in_backticks(text, 'json')
    if code_block:
        return json.loads(code_block)

    # NOTE: What if it's a list?
    json_start = text.find("{")
    if json_start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    json_str = text[json_start:]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if x.msg == 'Extra data':
            json_str = json_str[:e.pos]
            return json.loads(json_str)
        raise e
