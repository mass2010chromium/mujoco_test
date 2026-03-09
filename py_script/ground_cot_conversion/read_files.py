from openpi.policies.libero_reason_dataset import LiberoSkillReasonDataset
from openpi.training import config as _config

data_config = _config.get_config('pi05_libero_skill_reason_lora_v2')
dataset = LiberoSkillReasonDataset(data_config.data.base_config, data_config.model.action_horizon)

dataset[0]
