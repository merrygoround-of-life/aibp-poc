
from executor.core.agent import Agent, AgentConfig


class LocationAgent(Agent):
    description = "Use when you want to know where you're in."

    def __init__(self, config=AgentConfig(), name: str = None):
        super().__init__(config=config, name=name)

    def execute(self, agent_input: dict) -> dict:
        return {"location": "Seoul"}
