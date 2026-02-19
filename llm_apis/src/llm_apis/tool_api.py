"""
Shared interface for making tool calls to models via different LLM APIs.
"""

import inspect
from functools import wraps
import re
import time
import textwrap

try:
    from agent_responses import StringResponse, IntResponse, ImageResponse
    import transformers_api
except ImportError:
    from .agent_responses import StringResponse, IntResponse, ImageResponse
    from . import transformers_api


def tool_wrap(response_type=StringResponse, role: str = 'assistant'):
    """
    Decorator for wrapping tool execution.
    """
    def func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return response_type(f(*args, **kwargs), role=role)
        return wrapper
    return func


class ToolHandler:

    def __init__(self):
        self.custom_methods = {}
        self.message_state = []

    def register(self, f):
        self.custom_methods[f.__name__] = f

    def __getattr__(self, name):
        if name in self.custom_methods:
            return self.custom_methods[name]
        raise AttributeError(f"No custom function named `{name}`")

    # TODO: performance
    def match_tool_call(self, text):
        pattern = r"```tool_code\s*(.*?)\s*```"
        return re.search(pattern, text, re.DOTALL)

    def run_tool_call(self, matched):
        code = matched.group(1).strip()
        # Capture stdout in a string buffer
        if '\n' in code:
            return StringResponse('Tool call Error: Only one tool call can be processed at a time')
        try:
            result = eval("self."+code)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return StringResponse(repr(e))
        return result

    def get_tool_prompt(self):
        # TODO: cache?
        func_descriptions = []
        for fname, f in self.custom_methods.items():
            full_desc = inspect.getsource(f)
            matched = re.search('def .*"""|def .*\'\'\'', full_desc, re.DOTALL)
            if matched is None:
                print(full_desc)
                raise SyntaxError(f"Function definition missing docstring: {fname}")
            func_descriptions.append(full_desc[matched.start():matched.end()])
        function_str = "\n\n".join(func_descriptions)
        return self.get_init_prompt(function_str)

    def get_init_prompt(self, function_str=''):
        # Gemma 3 tool prompt
        return textwrap.dedent('''\
            At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```.
            The python methods described below are imported and available, you can only use defined methods.
            The generated code should be readable and efficient. The response to a method will be wrapped in
            ```tool_output```; use it to call more tools or generate a helpful, friendly response.
            When using a ```tool_call``` think step by step why and how it should be used.
            For example, if the function `echo` is defined, you can use it as follows:

            ```tool_code
            echo("testing!")
            ```

            DO NOT ASSIGN VARIABLES! Outputs from called functions will be fed back to you as JSON and image data directly.
            You may only call one function at a time. Attempting to call multiple functions or adding comments to
            the function call will result in an error.

            The following Python methods are available:
             
            ```python
            {0}
            ```
            '''
        ).format(function_str)

    def reset_chat(self, system_prompt):
        tool_prompt = self.get_tool_prompt()
        print(tool_prompt)
        self.message_state = [
            {'role': 'system', 'content': tool_prompt},
            {'role': 'system', 'content': system_prompt},
        ]


    def tool_chat(self, *args, **kwargs):
        """
        Main entry point for tool calling.
        """
        raise NotImplementedError("Base ToolHandler does not have implemented `tool_chat`")


    def add_message(self, text: str = '', images_rgb = [], role: str = 'user'):
        """
        Add a message to the message history.
        Abstracts exact message internal format
        """
        # Default: ollama format
        self.message_state.append({
            'role': role,
            'content': text,
            'images': [ ImageResponse.encode_image(img) for img in images_rgb ]
        })


class OllamaToolHandler(ToolHandler):
    """
    Tool handler specialization for using ollama python API.
    Takes in client and kwargs as part of `tool_chat` instead of constructor,
    because that's functionally when the model is loaded, and it might be resized based on kwargs.
    """

    def tool_chat(self, client, **kwargs):
        messages = self.message_state
        print(messages[-1])
        while True:
            stream = client.chat(stream=True, messages=messages, **kwargs)
            llm_msg = ""
            chunk_done = False
            tool_match = None
            for chunk in stream:
                chunk_done = chunk.done
                if chunk.message.content:
                    llm_msg += chunk.message.content
                    # Awful performance (n^2)
                    tool_match = self.match_tool_call(llm_msg)
                    if tool_match:
                        break
                else:
                    print("??? no content in chunk")
                    # TODO: ???
            stream.close()
            # TODO: fix token counter
            #print(resp.prompt_eval_count, "tokens.")
            print("==================\n")
            if len(messages) > 30:
                print("Pruning old images...")
                n = 0
                for msg in messages[-30:-20]:
                    if 'images' in msg:
                        n += 1
                        del msg['images']
                print(f"Pruned {n} images.")

            if tool_match:
                print("------> Matched valid tool call")
                trunc_msg = llm_msg[:tool_match.end()]
                if len(trunc_msg) < len(llm_msg):
                    print(f"WARNING: Wasting compute! Truncated message length {len(llm_msg)} -> {len(trunc_msg)}")
                print(trunc_msg)
                tool_resp = self.run_tool_call(tool_match)
                # The llm really likes to call the function multiple times without
                # waiting for a response
                messages.append({'role': 'assistant', 'content': trunc_msg})
                tool_msg = tool_resp.to_msg()
                messages.append(tool_msg)
            # For some reason it likes to set the done flag when calling functions
            elif chunk_done:
                if chunk.done_reason != 'stop':
                    print(f"Strange done_reason: {chunk.done_reason}")
                    time.sleep(1)
                    continue
                if len(llm_msg.strip()) == 0:
                    print("Got empty message, trying again")
                    messages.append({'role': 'user', 'content': "Go on..."})
                else:
                    messages.append({'role': 'assistant', 'content': llm_msg})
                    return llm_msg
            else:
                print(llm_msg)
                messages.append({'role': 'assistant', 'content': llm_msg})


class TransformersToolHandler(ToolHandler):
    """
    Huggingface Transformers API tool call handler.
    """

    def __init__(self, model, processor):
        super().__init__()
        self.prior_text = ""
        self.prior_images = []
        self.model = model
        self.processor = processor

    def get_init_prompt(self, function_str=''):
        # R4B tool prompt
        return textwrap.dedent('''\
            At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```.
            The python methods described below are imported and available, you can only use defined methods.
            The generated code should be readable and efficient. The response to a method will be wrapped in
            ```tool_output```; use it to call more tools or generate a helpful, friendly response.
            For example, if the function `echo` is defined, you can use it as follows:

            ```tool_code
            echo("testing!")
            ```

            DO NOT ASSIGN VARIABLES! Outputs from called functions will be fed back to you as JSON and image data directly.
            You may only call one function at a time. Attempting to call multiple functions or adding comments to
            the function call will result in an error.

            Function calls cannot be used in thinking mode. You cannot call functions in think mode. Finish thinking
            before calling any functions.

            The following Python functions are available:

            ```python
            {0}
            ```
            '''
        ).format(function_str)

    def reset_chat(self, system_prompt):
        super().reset_chat(system_prompt)
        self.prior_text = ''
        self.prior_images = []

    @staticmethod
    def convert_messages(message_state):
        """
        Convert a list of "ollama style" messages into Transformers style.
        TODO: support video.
        """
        messages = []
        for message in message_state:
            texts = []
            text_content = message.get('content', None)
            if text_content:
                texts.append(text_content)
            messages.append(transformers_api.make_message(
                texts=texts,
                images=message.get('images', []),
                role=message['role']
            ))
        return messages

    
    def tool_chat(self, block_size=512, max_new_tokens=16384, **kwargs):
        # NOTE: old image pruning is not supported. Context might grow to be too huge
        generated_tokens = 0
        while True:
            messages = TransformersToolHandler.convert_messages(self.message_state)
            llm_msg, text_context, image_context, eos, num_new_tokens = transformers_api.generate_output(
                self.model, self.processor, messages, self.prior_text, self.prior_images,
                max_new_tokens=block_size, **kwargs
            )
            self.prior_images = image_context
            self.message_state = []

            generated_tokens += num_new_tokens
            print(llm_msg, end='', flush=True)
            if generated_tokens >= max_new_tokens:
                self.prior_text = text_context + "</think>\nResponse Error.\n"
                print("WARNING: Stopped generation due to running out of tokens. Rolling back internal state")
                return llm_msg

            while True:
                done_thinking_tag = re.search('</think>', llm_msg, re.DOTALL)
                if done_thinking_tag:
                    thinking_end = done_thinking_tag.end()
                    tool_match = self.match_tool_call(llm_msg[thinking_end:])
                    if tool_match or eos:
                        break
                more_llm_msg, _, _, eos, num_new_tokens = transformers_api.generate_output(
                    self.model, self.processor, [], text_context + llm_msg, self.prior_images,
                    max_new_tokens=block_size, **kwargs
                )
                print(more_llm_msg, end='', flush=True)
                llm_msg += more_llm_msg

                generated_tokens += num_new_tokens
                if generated_tokens >= max_new_tokens:
                    self.prior_text = text_context + "</think>\nResponse Error.\n"
                    print("WARNING: Stopped generation due to running out of tokens. Rolling back internal state")
                    return llm_msg
            print()

            if not tool_match:
                self.prior_text = text_context + llm_msg
                return llm_msg.split('</think>', 1)[1]
            
            trunc_msg = llm_msg[:thinking_end + tool_match.end()]
            self.prior_text = text_context + trunc_msg
            print("------> Matched valid tool call")
            if len(trunc_msg) < len(llm_msg):
                print(f"WARNING: Wasting compute! Truncated message length {len(llm_msg)} -> {len(trunc_msg)}")
            tool_resp = self.run_tool_call(tool_match)
            tool_msg = tool_resp.to_msg(raw=True)
            self.message_state.append(tool_msg)


    def add_message(self, text: str = '', images_rgb = [], role: str = 'user'):
        """
        Add a message to the message history.
        Abstracts exact message internal format
        """
        # Transformers format: image stays as raw data. No need to send over the wire
        self.message_state.append({
            'role': role,
            'content': text,
            'images': images_rgb
        })
