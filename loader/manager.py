import json

from executor.core.agent import Agent
from loader.serialize import to_dict
from loader.deserialize import from_dict


class AgentManager:
    def __init__(self):
        self.db: dict[str, str] = {}

    def _set(self, agent: Agent):
        self.db[agent.name] = json.dumps(to_dict(agent))

    def set(self, agent_data: dict):
        agent = from_dict(agent_data)
        self.db[agent.name] = json.dumps(agent_data)

    def get(self, agent_name: str) -> Agent | None:
        if agent_name in self.db:
            return from_dict(json.loads(self.db[agent_name]))

        return None
