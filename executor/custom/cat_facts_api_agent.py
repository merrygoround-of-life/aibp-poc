import requests
from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput


class CatFactsApiInput(AgentInput):
    limit: int = Field(description="limit the amount of results returned")
    max_length: int = Field(description="maximum length of returned fact")


class CatFactsApiAgent(Agent):
    description = "Use when you want to know the facts about the cat."

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def parameters_schema(self) -> dict:
        return CatFactsApiInput.schema()

    def invoke(self, user: str, agent_input: dict) -> dict:
        return {
            "response": str(requests.get("https://catfact.ninja/facts", params=agent_input).json())
        }
