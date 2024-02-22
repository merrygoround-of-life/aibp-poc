from datetime import datetime

from executor.core.agent import Agent, AgentConfig


class CurrentTimeAgent(Agent):
    description = "Use when you want to know the current time."

    def __init__(self, config=AgentConfig(), name: str = None):
        super().__init__(config=config, name=name)

    def execute(self, agent_input: dict) -> dict:
        return {"timestamp":  datetime.now().isoformat()}
