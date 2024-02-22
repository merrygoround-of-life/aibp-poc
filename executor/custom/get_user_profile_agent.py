from executor.core.agent import Agent, AgentConfig, AgentInput


class UserProfileAgent(Agent):
    description = "Use when you want to know your profile."

    def __init__(self, config=AgentConfig(), name: str = None):
        super().__init__(config=config, name=name)

    def execute(self, agent_input: dict) -> dict:
        return {"user":  "isaac", "company": "skt"}
