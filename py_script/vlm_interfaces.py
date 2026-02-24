import sys

from llm_apis.llm_tool import LLMTool

def get_openrouter_interfaces():
    """
    Set up VLM and LLM models via an OpenRouter interface.
    """
    import os
    from llm_apis.llm_tool import OpenRouterTool
    print("Using OpenRouter")
    VLM_MODEL = "google/gemini-2.5-pro"
    LLM_MODEL = "google/gemini-2.5-flash"

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        print('  export OPENROUTER_API_KEY="sk-or-v1-..."')
        sys.exit(1)

    print(f"  LLM: {LLM_MODEL}")
    print(f"  VLM: {VLM_MODEL}")
    llm_interface = LLMTool.make_factory(OpenRouterTool, LLM_MODEL, api_key, temperature=0.0)
    vlm_interface = LLMTool.make_factory(OpenRouterTool, VLM_MODEL, api_key, temperature=0.0)
    return llm_interface, vlm_interface

def get_ollama_interfaces():
    """
    Set up VLM and LLM models via the Ollama interface.
    """
    import requests
    from llm_apis.llm_tool import OllamaTool
    from ollama import Client
    print("Using Ollama")

    base_url = "http://localhost:11434"
    VLM_MODEL = "gemma3:27b"
    LLM_MODEL = "gemma3:27b"
    context_length = 2400

    if requests.get(base_url).status_code == 200:
        client = Client(host=base_url)
        api_return = client.list()
        avail_models = [model['model'] for model in api_return['models']]
    else:
        print("ERROR: Ollama is not running")
        sys.exit(1)

    if LLM_MODEL not in avail_models:
        print(f"ERROR: Model {LLM_MODEL} is not available (out of {avail_models})")
        sys.exit(2)
    if VLM_MODEL not in avail_models:
        print(f"ERROR: Model {VLM_MODEL} is not available (out of {avail_models})")
        sys.exit(3)

    print(f"  LLM: {LLM_MODEL}")
    print(f"  VLM: {VLM_MODEL}")
    llm_interface = LLMTool.make_factory(OllamaTool, client, model=LLM_MODEL, keep_alive=-1, 
                                         options=dict(num_ctx=context_length, temperature=0.1))
    vlm_interface = LLMTool.make_factory(OllamaTool, client, model=VLM_MODEL, keep_alive=-1, 
                                         options=dict(num_ctx=context_length, temperature=0.1))
    return llm_interface, vlm_interface


def get_r4b_interfaces():
    """
    Set up VLM and LLM models via Huggingface Transformers.
    Only tested for R4B. Likely to break for others due to thinking mode arguments
    """
    from llm_apis.llm_tool import TransformersTool
    from transformers import AutoModel, AutoProcessor
    import torch
    print("Using Transformers API")

    model_path = "YannQi/R-4B"

    # Load model
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to("cuda")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"  VLM (and LLM): {model_path}")
    llm_interface = LLMTool.make_factory(TransformersTool, model, processor)
    vlm_interface = llm_interface
    return llm_interface, vlm_interface
