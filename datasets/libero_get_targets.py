#!/usr/bin/env python3
"""
Process bddl files and see what the "target object" should be

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python libero_get_targets.py <libero_dataset_name>
"""

import time

import mediapy

from llm_apis.llm_tool import LLMTool
from llm_apis import transformers_api
from llm_apis.response_parsing import extract_json_from_response

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'py_script'))
from vlm_interfaces import *    # get_*_interfaces() functions for llm calling

def _analyze_bddl(llm_response, bddl_raw):
    t0 = time.monotonic()
    print(f"  [VLM] Querying VLM...")
    formatted = f'```bddl\n{bddl_raw}\n```'
    yield [ transformers_api.make_message(texts=[formatted]) ]
    t1 = time.monotonic()
    print(f"  [VLM] Time elapsed: {t1 - t0}")

    yield extract_json_from_response(llm_response['content'])

_analyze_bddl.system_prompt = """\
Given a bddl file (an extension of pddl, with labels for different regions of space), \
determine objects a robot interacting with this space should reach for to solve \
the task. The task is specified as a natural language goal, in the (:language) tag.

The natural language input may not correspond exactly with the object labels in the file. \
If that is the case, you should use the spatial context (such as the region information \
and coordinates) to disambiguate which object is the target of the instruction.

The (:obj_of_interest) tag may be helpful for your search.

Output a JSON object as your result, with the following format:

```json
{
  "plan": <description of the steps the robot would have to take to complete the task>,
  "target_objects": [
    {
      "id": <object id, matching the (:objects) tag>,
      "location": <region id, or "NULL" if no region matches
                    (or if the location is the just the object's location).
                    Use the full id (accounting for the (:target) tag:
                    For example, wooden_cabinet_1_top_region instead of top_region.>
    }
  ]
}
```
"""

def main():
    import argparse
    import json
    import os
    import libero
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name (ex. libero_10)")
    parser.add_argument("--resume", action='store_true', help="Don't rerun LLM queries if they already exist")
    args = parser.parse_args()

    libero_root = Path(libero.__file__).parent
    info_file = libero_root / 'libero' / 'bddl_files' / args.dataset / 'tasks_info.txt'
    target_files = []
    with open(info_file, 'r') as f:
        target_files = [s.strip() for s in f.readlines()]

    llm_interface, vlm_interface = get_openrouter_interfaces()
    #llm_interface, vlm_interface = get_ollama_interfaces()
    #llm_interface, vlm_interface = get_r4b_interfaces()

    # vlm is gemini pro
    analyze_bddl = vlm_interface(_analyze_bddl)

    output_dir = Path(__file__).parent / 'libero_targets'
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"{args.dataset}.json"
    all_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        except:
            pass
    for i, target_file in enumerate(target_files):
        if args.resume and str(i) in all_results:
            print(f"========== Skipping file {i} ==========")
            continue
        print(f"========== Processing file {i} ==========")
        print(target_file)
        with open(libero_root / target_file, 'r') as f:
            file_data = f.read()
        result = analyze_bddl(file_data)
        all_results[str(i)] = result
        print(result)
        print()
        all_results[str(i)]['task_name'] = target_file.rsplit('/', 1)[1].split('.', 1)[0]
        with open(output_file, 'w') as f:
            json.dump(all_results, f)



if __name__ == "__main__":
    main()
