#!/usr/bin/env python3
"""
Test script for calling VLMs

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python run_vlm.py
"""

import time

import mediapy

from llm_apis.llm_tool import LLMTool
from llm_apis import transformers_api
from llm_apis.response_parsing import extract_json_from_response

from vlm_interfaces import *

def _analyze_image(llm_response, image_rgb):
    t0 = time.monotonic()
    print(f"  [VLM] Querying VLM...")
    yield [ transformers_api.make_message(images=[image_rgb]) ]
    t1 = time.monotonic()
    print(f"  [VLM] Time elapsed: {t1 - t0}")

    yield extract_json_from_response(llm_response['content'])

_analyze_image.system_prompt = f"""\
Give the segmentation masks for relevant objects in the image. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the mask in "mask", and the text label in the key "label". 
Use descriptive labels.
"""


def main():

    llm_interface, vlm_interface = get_openrouter_interfaces()
    #llm_interface, vlm_interface = get_ollama_interfaces()
    #llm_interface, vlm_interface = get_r4b_interfaces()

    analyze_image = vlm_interface(_analyze_image)

    #IMAGE_PATH = 'image.png'
    IMAGE_PATH = '../franka_libero_init.png'
    #IMAGE_PATH = '../not_libero.png'
    import cv2
    import numpy as np
    img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    #img = mediapy.read_image(IMAGE_PATH)
    #objects = ['bowl', 'stove_heater']
    res = analyze_image(cv2.resize(img, (1024, 1024)))
    print(res)

    H, W = img.shape[:2]
    GEMINI_PX = 1024
    def box_to_img(box_xyxy):
        return np.array([box_xyxy[1]*W, box_xyxy[0]*H, box_xyxy[3]*W, box_xyxy[2]*H]) / GEMINI_PX

    for entry in res:
        img_box = box_to_img(entry['box_2d'])
        color = 255 * ((np.random.random(3) / 2) + 0.5)
        cv2.rectangle(img, img_box[:2].astype(int), img_box[2:].astype(int), color, thickness=2)

    mediapy.write_image(IMAGE_PATH+'.out', img)


if __name__ == "__main__":
    main()
