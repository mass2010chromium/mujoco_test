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

from llm_apis.llm_tool import TransformersTool, extract_json_from_response
from llm_apis import transformers_api

prompt = """
You are a calculator assistant.
The user will give you a calculation problem, you should return a JSON response of the following form:

{
    "result": <result as string>,
    "explanation": <explanation in english>
}
"""
def _calculate(llm_response, input_string):
    
    # These functions will be funny. They "return" twice.
    # The first time, the yielded value must be a List of messages, of the form:
    # {
    #   "role": <role>,
    #   "content": [
    #     {"type": "image"|"text", ...},
    #     {"type": "text", "text": "hello"},
    #     {"type": "image", "image": <image_rgb>}
    #   ]
    # }
    # (Yes, each message itself has a list of contents. Deal with it.)
    # This is fed to the LLM, which returns its response via the first function argument,
    #   which is a dictionary of the form
    # {
    #   "content": <text_content>
    # }
    # When this function gets "compiled", the first argument is hidden from the end user.
    # So it is called only with its second argument. See the invocations below

    # First return: Yield the list of messages (just one string message today)
    yield [ transformers_api.make_message(input_string) ]

    # Process output from LLM, and return to user
    raw_response = llm_response['content']
    yield extract_json_from_response(raw_response)
# Set the "system prompt" for this LLM.
_calculate.system_prompt = prompt

calculate = TransformersTool(_calculate, model, processor)

print(calculate("Whats 1+1?"))
print(calculate("Give a rational approximation of Pi"))
