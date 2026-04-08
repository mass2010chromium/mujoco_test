import json
import gzip

from openpi.policies.libero_reason_dataset import LiberoSkillReasonDataset
from openpi.training import config as _config

data_config = _config.get_config('pi05_libero_skill_reason_fixed')

episode_data = json.load(open(data_config.data.base_config.reasoning_json_path))
episodes = []
i = 0
while True:
    if str(i) not in episode_data:
        break
    episodes.append(episode_data[str(i)])
    i += 1

out_dir = "data"
out_data = []
reasons = []
n_fails = 0
for i in range(len(episodes)):
    with gzip.open(f"{out_dir}/{i}_targets.json.zip", "rt", encoding="ascii") as zipfile:
        target_data = json.load(zipfile)
    episode = episodes[i]
    episode_failed = False
    for j, segment in enumerate(episode['segments']):
        if j < len(target_data):
            target_info = target_data[j]
            segment['target'] = target_info
            if target_info is None:
                episode_failed = True
                reasons.append(f"{i}: Step {j} has no target info due to validation failure")
            elif 'image_point' not in target_info:
                episode_failed = True
                reasons.append(f"{i}: Step {j} has target info but no grounding point")
        else:
            segment['target'] = None
            if not episode_failed:
                episode_failed = True
                reasons.append(f"{i}: Length ({len(target_data)}) shorter than expected ({len(episode['segments'])})")
    if episode_failed:
        n_fails += 1

print(n_fails, "failures.")
with open("targets_log.txt", "w") as outfile:
    print('\n'.join(reasons), file=outfile)

with open("cot_targets.json", 'w') as outfile:
    json.dump(episode_data, outfile, indent=4)
