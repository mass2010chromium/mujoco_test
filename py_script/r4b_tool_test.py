import requests
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from tool_api import R4BToolHandler
from agent_responses import tool_wrap, ImageResponse

model_path = "YannQi/R-4B"

# Load model
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

tool_handler = R4BToolHandler(model, processor)

def web_request(url):
    """
    Make a web request.

    @param url  URL to request

    @return Image, if the url is an image file, or String for a webpage or error message.
    """
    allowed_urls = [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        'http://images.cocodataset.org/train2017/000000220266.jpg',
        'http://images.cocodataset.org/train2017/000000141955.jpg'
    ]
    if url in allowed_urls:
        print("Making web request...")
        image = Image.open(requests.get(url, stream=True).raw)
        return ImageResponse(img=image)
    print("Bad URL")
    return StringResponse("Could not access URL...")

tool_handler.register(web_request)

tool_handler.reset_chat("You are an AI assistant with a web search tool.")

def chat(user_message):
    tool_handler.message_state.append({
        'role': 'user', 'content': user_message
    })
    return tool_handler.tool_chat()


