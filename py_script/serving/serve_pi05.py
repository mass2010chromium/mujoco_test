import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

from typing import Dict

from inference_common import load_policy
# Not used in this file, but import to register the models.
import pi05_action
import pi05_plan_skill_action
import pi05_plan_transition_action

from openpi_client.base_policy import BasePolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

class ServePolicy(BasePolicy):
    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs: Dict) -> Dict:
        return self._policy.run_vla(obs)

    def initialize(self, obs: Dict, task: str) -> None:
        self._policy.initialize(obs, task)

    def reset(self) -> None:
        self._policy.reset()

policy = load_policy('pi05_real_lora')

server = WebsocketPolicyServer(policy=ServePolicy(policy), port=9898)
server.serve_forever()
