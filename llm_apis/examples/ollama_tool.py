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
    keep_alive=0,
    options=dict(
        num_ctx=context_length,
        temperature=0.0
    )
)
print(resp)
