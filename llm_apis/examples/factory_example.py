from llm_apis.llm_tool import LLMTool, json_output
from llm_apis import transformers_api

def get_ollama_interface():
    from llm_apis.llm_tool import OllamaTool
    from ollama import Client

    base_url="http://localhost:11434"  # Running locally
    model_name="gemma3:27b" # Change your model here
    context_length = 2400   # Change max context length
    client = Client(host=base_url)

    return LLMTool.make_factory(OllamaTool, client, model=model_name, keep_alive=0,
                                options=dict(num_ctx=context_length, temperature=0.1))

def get_r4b_interface():
    from llm_apis.llm_tool import TransformersTool
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

    return LLMTool.make_factory(TransformersTool, model, processor)

#llm_interface = get_ollama_interface()
llm_interface = get_r4b_interface()

# Factories can be used with the json_output decorator!
@json_output
def _guess_mood(input_string):
    return [ transformers_api.make_message(input_string) ]

_guess_mood.system_prompt = """\
You are a sentiment analysis assistant.
Given the piece of text, determine what the overall sentiment is, and give reasoning.
Return your response as json, in this form:

{
  "sentiment": <sentiment_string>,
  "confidence": <float, 0 to 1>,
  "reasoning": <reasoning>
}
"""

guess_mood = llm_interface(_guess_mood)
print(guess_mood("The sky seems sunny and bright today!"))
