import requests
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_path = "YannQi/R-4B"

# Load model
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
image2_url = "http://images.cocodataset.org/train2017/000000220266.jpg"
image2 = Image.open(requests.get(image2_url, stream=True).raw)

# Define conversation messages
messages = [
    {
        "role": "user",
        "content": [
            {'type': 'text', 'text': 'You are a robotic assistant translating human queries into JSON query results. Return your response in the following form (including the triple backticks and json indicator):\n\n```json\n{\n  "result": <result>\n}\n```\n\n'},
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "image",
                "image": image2,
            },
            {"type": "text", "text": "Which image has more cats? The first or second one? Can you explain your reasoning?"},
        ],
    },
]

# Apply chat template
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    thinking_mode="auto"
)
print(text)

# Process inputs
inputs = processor(
    images=[image, image2],
    text=text,
    return_tensors="pt"
).to("cuda")


# Generate output

tokenizer = AutoTokenizer.from_pretrained(model_path)

def stream_output(inputs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_args = {
        **inputs,
        'max_new_tokens': 16384,
        'streamer': streamer
    }
    thread = Thread(
        target=model.generate,
        kwargs=generation_args
    )
    thread.start()

    all_output = ''
    for text_token in streamer:
        all_output += text_token
        print(text_token, end='', flush=True)
    print()
    thread.join()
    return all_output

all_output = stream_output(inputs)

text += all_output
image3_url = "http://images.cocodataset.org/train2017/000000141955.jpg"
image3 = Image.open(requests.get(image3_url, stream=True).raw)
text += processor.apply_chat_template(
    [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'You are a robotic assistant translating human queries into JSON query results. Return your response in the following form (including the triple backticks and json indicator):\n\n```json\n{\n  "result": <result>\n}\n```\n\n'},
                {'type': 'image', 'image': image3},
                {'type': 'text', 'text': 'I got one more image. Now how many cats are there in total?'}
            ]
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
    thinking_mode="auto"
)
print(text)

# Process inputs
inputs = processor(
    images=[image, image2, image3],
    text=text,
    return_tensors="pt"
).to("cuda")

all_output = stream_output(inputs)

# # Decode output
# generated_ids = model.generate(**inputs, max_new_tokens=16384)
# output_ids = generated_ids[0][len(inputs.input_ids[0]):]
# output_text = processor.decode(
#     output_ids,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=False
# )
# 
# # Print result
# print("Auto-Thinking Output:", output_text)

