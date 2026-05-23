import math
import os
import pathlib

import numpy as np

import openpi
OPENPI_ROOT = os.path.join(os.path.dirname(os.path.abspath(openpi.__file__)), '..', '..')
from openpi.shared import download
from openpi.training import config as _config

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def get_model_info(model_name, checkpoint_num, run_name=None, checkpoint_dir=None):
    if run_name is None:
        run_name = model_name
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(OPENPI_ROOT, 'checkpoints', model_name, run_name, str(checkpoint_num))
    assets_dir = os.path.join(OPENPI_ROOT, 'assets', model_name)
    vla_config = _config.get_config(model_name)
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

    # Load norm stats from assets dir if checkpoint doesn't have them
    norm_stats = None
    data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
    if data_config.asset_id is not None:
        from openpi.training import checkpoints as _checkpoints
        assets_path = pathlib.Path(assets_dir).resolve()
        norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)
    return vla_config, checkpoint_dir, norm_stats


MODEL_REGISTRY = {}

def register_base_model(factory_func, model_name: str):
    """
    Register a model from physical intelligence.
    """
    def create_model():
        return factory_func(model_name)
    MODEL_REGISTRY[model_name] = {0: create_model}

def register_model(factory_func, model_name: str, checkpoint_num: int, run_name: str = None, _checkpoint_dir: str = None):
    """
    Register a locally downloaded model.
    """
    if model_name not in MODEL_REGISTRY:
        MODEL_REGISTRY[model_name] = {}
    models = MODEL_REGISTRY[model_name]

    def create_model():
        config, checkpoint_dir, norm_stats = get_model_info(model_name, checkpoint_num, run_name=run_name, checkpoint_dir=_checkpoint_dir)
        return factory_func(model_name, config, checkpoint_dir, norm_stats)
    models[checkpoint_num] = create_model


def load_policy(model_name, checkpoint_num=None):
    models = MODEL_REGISTRY[model_name]
    if checkpoint_num is None:
        checkpoint_num = sorted(models.keys())[-1]
    print(f"Loading model {model_name}/{checkpoint_num}...")
    return models[checkpoint_num]()
