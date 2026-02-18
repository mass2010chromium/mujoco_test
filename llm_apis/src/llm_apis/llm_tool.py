import json
import re
import textwrap

import requests

try:
    from agent_responses import StringResponse, IntResponse, ImageResponse
    import transformers_api
except ImportError:
    from .agent_responses import StringResponse, IntResponse, ImageResponse
    from . import transformers_api

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from a VLM response that may contain markdown fences."""
    # Strip markdown code fences if present
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_block:
        json_str = code_block.group(1).strip()
        return json.loads(json_str)

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

def dict_to_json_str(indent=4):
    """Basic formatter to dump the first positional argument as a json dictionary."""

    def format_kwargs(obj):
        return textwrap.dedent(f"""\
            ```json
            {json.dumps(obj, indent=indent)}
            ```
            """
        )
    return format_kwargs


class JsonTool:
    def __init__(self, system_prompt, function, system_prompt_role='system'):
        self.system_prompt = system_prompt
        self.system_prompt_role = system_prompt_role
        # Function should return a list of transformers style messages
        # {
        #   "role": <role>,
        #   "content": [
        #     {"type": "image"|"text", ...},
        #     {"type": "text", "text": "hello"},
        #     {"type": "image", "image": <image_rgb>}
        #   ]
        # }
        self.function = function

    def get_system_prompt_message(self):
        return {
            'role': self.system_prompt_role,
            'content': [
                {'type': 'text', 'text': self.system_prompt}
            ]
        }

    def __call__(self, *args, **kwargs):
        messages = [self.get_system_prompt_message()] + self.function(*args, **kwargs)
        response = self.make_query(messages)
        return extract_json_from_response(response)

    def make_query(self, messages):
        raise NotImplementedError("make_query should be specialized per llm provider")

class TransformersJsonTool(JsonTool):

    def __init__(self, system_prompt, function, model, processor, system_prompt_role='system', max_new_tokens=16384):
        super().__init__(system_prompt, function, system_prompt_role)
        # TODO: pass other kwargs
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def make_query(self, messages):
        text, _, _, _, _  = transformers_api.generate_output(self.model, self.processor, messages, max_new_tokens=self.max_new_tokens)
        return text


class OpenRouterJsonTool(JsonTool):

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, system_prompt, function, model, api_key,
                 system_prompt_role='system', temperature: float = 0.2, timeout: int = 180):
        super().__init__(system_prompt, function, system_prompt_role)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def make_query(self, messages):
        for turn in messages:
            for message in turn:
                if message['type'] == 'image':
                    message['type'] = 'image_url'

                    image = message['image']

                    base64_data = ImageResponse.encode_image(image_rgb)
                    mime = "image/png"
                    data_url = f"data:{mime};base64,{base64_data}"
                    message['image_url'] = {
                        "url": data_url
                    }

                    del message['image']

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            OpenRouterJsonTool.OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"Unexpected VLM response: {data}")

        # TODO: only string responses supported
        return choices[0].get("message", {}).get("content", "").strip()
