import requests
from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput


class DogFactsApiInput(AgentInput):
    number: int = Field(description="limit the amount of results returned")


class DogFactsApiAgent(Agent):
    description = "Use when you want to know the facts about the dog."

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def parameters_schema(self) -> dict:
        return DogFactsApiInput.schema()

    def invoke(self, user: str, agent_input: dict) -> dict:
        return {
            "response": str(requests.get("https://dog-api.kinduff.com/api/facts", params=agent_input).json())
        }
