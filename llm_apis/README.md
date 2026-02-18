# Unified LLM query api

Ways to more easily call LLMs, with a unified interface.

Call LLMs locally (or network) through ollama, or with huggingface Transformers api, or with openrouter.

Support is uneven and implemented on an as-needed basis...

## Install

```bash
# In repository root
pip install -e .
```

## Example tool caller

Example is given using Ollama. Currently, Ollama and HuggingFace/Transformers api supported.

```python
from typing import Optional
from llm_apis.tool_api import OllamaToolHandler, tool_wrap
from llm_apis.agent_responses import StringResponse

# Note that the docstring is mandatory for tool calling functions!
@tool_wrap(StringResponse)  # Wraps the output into a response message type
def get_custom_greeting(user: Optional[str] = None) -> str:
    """
    Use this to get the preferred greeting for a given user.

    Parameters:
        user (str, Optional):   User's name
    
    Returns:
        String with custom greeting
    """
    return f"Hellllooooo, {user}!"

    # Optionally, if no tool_wrap:
    # Useful if you want to return custom error messages, for example
    # return StringResponse(f"Hellllooooo, {user}! but more customizable!", role='user')

tool_handler = OllamaToolHandler()

# Register custom tool
tool_handler.register(get_custom_greeting)

system_prompt = "You are a helpful assistant."

# Sets up tool prompt
tool_handler.reset_chat(system_prompt)

# Set up Ollama client
from ollama import Client

base_url="http://localhost:11434"  # Running locally
model_name="gemma3:27b" # Change your model here
context_length = 2400   # Change max context length
client = Client(host=base_url)

tool_handler.add_message("Hello! I'm Bob, how are you?")
resp = tool_handler.tool_chat(
    client,
    model=model_name,
    keep_alive=-1,
    options=dict(
        num_ctx=context_length,
        temperature=0.0
    )
)
print(resp)
```

## Example llm-as-function

Example is given using Transformers. Currently, OpenRouter and HuggingFace/Transformers api supported.

```python
import torch
from transformers import AutoModel, AutoProcessor

model_path = "YannQi/R-4B"

# Load model
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

from llm_apis.llm_tool import TransformersJsonTool
from llm_apis import transformers_api
prompt = """
You are a calculator assistant.
The user will give you a calculation problem, you should return a JSON response of the following form:

{
    "result": <result as string>,
    "explanation": <explanation in english>
}

The problem is:
"""

# Must return a List of messages
def _calculate(input_string):
    return [ transformers_api.make_message(input_string) ]

calculate = TransformersJsonTool(prompt, _calculate, model, processor)

print(calculate("Whats 1+1?"))
print(calculate("Give a rational approximation of Pi"))
```


