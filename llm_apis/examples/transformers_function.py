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
