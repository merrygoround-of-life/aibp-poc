import json

from executor.core.agent import Agent
from loader.serialize import to_dict
from loader.deserialize import from_dict


class AgentManager:
    def __init__(self):
        self.db: dict[str, str] = {}

    def _set(self, agent: Agent) -> str:
        self.db[agent.uuid] = json.dumps(to_dict(agent))
        return agent.uuid

    def set(self, agent_data: dict) -> str:
        agent = from_dict(agent_data)
        self.db[agent.uuid] = json.dumps(agent_data)
        return agent.uuid

    def get(self, agent_uuid: str) -> Agent | None:
        if agent_uuid in self.db:
            return from_dict(json.loads(self.db[agent_uuid]))

        return None
