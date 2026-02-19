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

from llm_apis.llm_tool import TransformersTool, json_output
from llm_apis import transformers_api

prompt = """
You are a calculator assistant.
The user will give you a calculation problem, you should return a JSON response of the following form:

{
    "result": <result as string>,
    "explanation": <explanation in english>
}
"""
@json_output
def _calculate(input_string):
    # Convenience annotation to not deal with llm response handshake.
    # Nice for anything that you just want in JSON.
    return [ transformers_api.make_message(input_string) ]
# Set the "system prompt" for this LLM.
_calculate.system_prompt = prompt

calculate = TransformersTool(_calculate, model, processor)

print(calculate("Whats 1+1?"))
print(calculate("Give a rational approximation of Pi"))
