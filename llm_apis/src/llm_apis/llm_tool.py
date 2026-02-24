import copy
import functools
import re
import textwrap

import requests

try:
    from .agent_responses import StringResponse, IntResponse, ImageResponse
    from . import transformers_api
    from .response_parsing import extract_json_from_response
except ImportError:
    from agent_responses import StringResponse, IntResponse, ImageResponse
    import transformers_api
    from response_parsing import extract_json_from_response

def json_output(f):
    """
    Decorator to make a function return JSON when used with LLMTool.
    The function should use normal arguments (no extra argument 0).
    """
    @functools.wraps(f)
    def wrapper(llm_response, *args, **kwargs):
        yield f(*args, **kwargs)
        yield extract_json_from_response(llm_response['content'])
    return wrapper

class LLMTool:
    def __init__(self, function, system_prompt=None, system_prompt_role='system'):
        # To define the system prompt, attach it to the function as function.system_prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = function.system_prompt
        # TODO: should this also be attached?
        self.system_prompt_role = system_prompt_role
        # The function should take one additional argument as the first position argument.
        # This will be a placeholder, for the llm response, as a dictionary:
        # {
        #   "content": <text_content>
        # }
        # Other arguments are the normal function call arguments.
        # Function should "return twice" via yield.
        # First time, a list of transformers style messages:
        # {
        #   "role": <role>,
        #   "content": [
        #     {"type": "image"|"text", ...},
        #     {"type": "text", "text": "hello"},
        #     {"type": "image", "image": <image_rgb>}
        #   ]
        # }
        # 
        # Second time, the actual output.
        self.function = function

    @staticmethod
    def make_factory(tool_class, *args, **kwargs):
        def _factory(function, **override_kwargs):
            _kwargs = copy.deepcopy(kwargs)
            _kwargs.update(override_kwargs)
            return tool_class(function, *args, **_kwargs)
        return _factory

    def get_system_prompt_message(self):
        return {
            'role': self.system_prompt_role,
            'content': [
                {'type': 'text', 'text': self.system_prompt}
            ]
        }

    def __call__(self, *args, **kwargs):
        llm_response = {}
        interactive_handle = self.function(llm_response, *args, **kwargs)

        messages = [self.get_system_prompt_message()] + next(interactive_handle)
        response = self.make_query(messages)
        self.raw_response = response
        llm_response['content'] = response
        return next(interactive_handle)

    def make_query(self, messages):
        raise NotImplementedError("make_query should be specialized per llm provider")



class OllamaTool(LLMTool):

    def __init__(self, function, client, model,
                 system_prompt=None,
                 system_prompt_role='system',
                 **kwargs):
        super().__init__(function, system_prompt, system_prompt_role)
        # TODO: pass other kwargs
        self.client = client
        self.model = model
        self.kwargs = kwargs

    def make_query(self, messages):
        ollama_messages = []
        for turn in messages:
            texts = []
            images_rgb = []
            for message in turn['content']:
                if message['type'] == 'text':
                    texts.append(message['text'])
                if message['type'] == 'image':
                    images_rgb.append(message['image'])
            all_text = '\n'.join(texts)
            ollama_messages.append({
                'role': turn['role'],
                'content': all_text,
                'images': [ ImageResponse.encode_image(img) for img in images_rgb ]
            })
        chat_response = self.client.chat(stream=False, messages=ollama_messages, model=self.model, **self.kwargs)
        return chat_response.message.content

class TransformersTool(LLMTool):

    def __init__(self, function, model, processor,
                 system_prompt=None,
                 system_prompt_role='system',
                 max_new_tokens=16384):
        super().__init__(function, system_prompt, system_prompt_role)
        # TODO: pass other kwargs
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def make_query(self, messages):
        llm_msg, _, _, _, _  = transformers_api.generate_output(self.model, self.processor, messages, max_new_tokens=self.max_new_tokens)
        done_thinking_tag = re.search('</think>', llm_msg, re.DOTALL)
        if done_thinking_tag:
            thinking_end = done_thinking_tag.end()
            return llm_msg[thinking_end:]
        print("WARNING: model did not finish thinking")
        return llm_msg


class OpenRouterTool(LLMTool):

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, function, model, api_key,
                 system_prompt=None,
                 system_prompt_role='system',
                 temperature: float = 0.2,
                 timeout: int = 180):
        super().__init__(function, system_prompt, system_prompt_role)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def make_query(self, messages):
        for turn in messages:
            for message in turn['content']:
                if message['type'] == 'image':
                    message['type'] = 'image_url'

                    image_rgb = message['image']

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
            OpenRouterTool.OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"Unexpected VLM response: {data}")

        # TODO: only string responses supported
        return choices[0].get("message", {}).get("content", "").strip()
