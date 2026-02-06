import inspect
import re
import time

try:
    from agent_responses import StringResponse, IntResponse, ImageResponse
    from agent_responses import tool_wrap
except ImportError:
    from .agent_responses import StringResponse, IntResponse, ImageResponse
    from .agent_responses import tool_wrap

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
        tool_prompt = f'''At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.
 
The following Python methods are available:
 
```python
{function_str}
```
'''
        return tool_prompt

    def reset_chat(self, system_prompt):
        tool_prompt = self.get_tool_prompt()
        print(tool_prompt)
        self.message_state = [
            {'role': 'system', 'content': tool_prompt},
            {'role': 'system', 'content': system_prompt},
        ]

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
                messages.append(tool_resp.to_msg())
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


