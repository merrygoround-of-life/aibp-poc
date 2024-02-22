import requests

from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput


class CatFactsInput(AgentInput):
    limit: int = Field(description="limit the amount of results returned")
    max_length: int = Field(description="maximum length of returned fact")


class CatFactsAgent(Agent):
    description = "Use when you want to know the facts about the cat."

    def __init__(self, config=AgentConfig(), name: str = None):
        super().__init__(config=config, name=name)

    def parameters_schema(self) -> dict:
        return CatFactsInput.schema()

    def execute(self, agent_input: dict) -> dict:
        return {"response": str(requests.get("https://catfact.ninja/facts", params=agent_input).json())}
