import json
import math

from openpi.policies.libero_reason_dataset import LiberoSkillReasonDataset
from openpi.training import config as _config

data_config = _config.get_config('pi05_libero_skill_reason_fixed')
dataset = LiberoSkillReasonDataset(data_config.data.base_config, data_config.model.action_horizon)

N_SPLITS = 8

N = len(dataset.episode_starts)
block = math.ceil(N / N_SPLITS)
with open("splits.json", 'w') as outfile:
    json.dump(list(range(0, N, block)), outfile)
