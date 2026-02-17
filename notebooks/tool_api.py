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
        return f'''At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:
 
```python
{function_str}
```
'''

    def reset_chat(self, system_prompt):
        tool_prompt = self.get_tool_prompt()
        print(tool_prompt)
        self.message_state = [
            {'role': 'system', 'content': tool_prompt},
            {'role': 'system', 'content': system_prompt},
        ]


    def tool_chat(self, *args, **kwargs):
        raise NotImplementedException("Base ToolHandler does not have implemented `tool_chat`")

class OllamaToolHandler(ToolHandler):

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

class R4BToolHandler(ToolHandler):
    def __init__(self, model, processor):
        super().__init__()
        self.prior_text = ""
        self.prior_images = []
        self.model = model
        self.processor = processor

    def get_init_prompt(self, function_str=''):
        # R4B tool prompt
        return f'''At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. For example, if the function `echo` is defined, you can use it as follows:

```tool_code
echo("testing!")
```

DO NOT ASSIGN VARIABLES! Outputs from called functions will be fed back to you as JSON and image data directly.
You may only call one function at a time. Attempting to call multiple functions or add comments to the function call will result in an error.

The following Python functions are available:

```python
{function_str}
```
'''

    def reset_chat(self, system_prompt):
        super().reset_chat(system_prompt)
        self.prior_text = ''
        self.prior_images = []

    @staticmethod
    def make_message(texts=[], images=[], role="user"):
        """
        Makes a message structure containing the appropriate information for the given data.
        """
        messages = []
        for image in images:
            messages.append({'type': 'image', 'image': image})
        for text in texts:
            messages.append({'type': 'text', 'text': text})
        return {"role": role, "content": messages}

    def _convert_messages(self):
        messages = []
        for message in self.message_state:
            texts = []
            text_content = message.get('content', None)
            if text_content:
                texts.append(text_content)
            messages.append(R4BToolHandler.make_message(
                texts=texts,
                images=message.get('images', []),
                role=message['role']
            ))
        return messages

    def _generate_output(self, messages, prior_text='', prior_images=[], thinking_mode='auto', device='cuda', max_tokens=200):
        """
        Generate some output from the R4B VLM.

        Messages are the normal content/type/role messages.
        Prior text/images are the previous context, since R4B doesn't have a clean way
        of adding generated output back to the messages list.
        
        Return: (response, new_text_context, new_image_context, eos)
        """
        import torch

        images = []
        for turn in messages:
            for message in turn['content']:
                if message['type'] == 'image':
                    images.append(message['image'])
    
        # This isn't quite tokenization, it just puts tags like <|im_start|> for turns or <think> to begin thinking
        if len(messages) > 0:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode=thinking_mode
            )
            text_context = prior_text + text
        else:
            text_context = prior_text
    
        #Concatenate text and image contexts
        image_context = prior_images+images
        inputs = self.processor(
            images=image_context if len(image_context) else None,
            text=text_context,
            return_tensors='pt'
        ).to(device)
    
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        eos_id = self.processor.tokenizer.eos_token_id
        eos_output = torch.any(output_ids == eos_id)
        output_text = self.processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text, text_context, image_context, eos_output
    
    def tool_chat(self, **kwargs):
        while True:
            messages = self._convert_messages()
            llm_msg, text_context, image_context, eos = self._generate_output(messages, self.prior_text, self.prior_images, **kwargs)
            self.prior_images = image_context
            self.message_state = []
            print(llm_msg, end='', flush=True)
            while True:
                done_thinking_tag = re.search('</think>', llm_msg, re.DOTALL)
                if done_thinking_tag:
                    thinking_end = done_thinking_tag.end()
                    tool_match = self.match_tool_call(llm_msg[thinking_end:])
                    if tool_match or eos:
                        break
                more_llm_msg, _, _, eos = self._generate_output([], text_context + llm_msg, self.prior_images, **kwargs)
                print(more_llm_msg, end='', flush=True)
                llm_msg += more_llm_msg
            print()

            if not tool_match:
                self.prior_text = text_context + llm_msg
                return llm_msg.split('</think>', 1)[1]
            
            trunc_msg = llm_msg[:thinking_end + tool_match.end()]
            self.prior_text = text_context + trunc_msg
            print(thinking_end, tool_match.end())
            print("------> Matched valid tool call")
            if len(trunc_msg) < len(llm_msg):
                print(f"WARNING: Wasting compute! Truncated message length {len(llm_msg)} -> {len(trunc_msg)}")
            tool_resp = self.run_tool_call(tool_match)
            self.message_state.append(tool_resp.to_msg(raw=True))
