from datetime import datetime

from executor.core.agent import Agent, AgentConfig


class CurrentTimeAgent(Agent):
    description = "Use when you want to know the current time."

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def invoke(self, user: str, agent_input: dict) -> dict:
        return {"timestamp": datetime.now().isoformat()}
