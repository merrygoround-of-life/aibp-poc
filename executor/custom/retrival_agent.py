from pydantic import Field

from executor.core.agent import AgentType, AgentInput
from executor.core.chain_agent import ChainAgent


class RetrievalAgentInput(AgentInput):
    prompts: list[str] = Field(description="a prompt list for user's family members who want to know")


class RetrievalAgent(ChainAgent):
    type = AgentType.CHAIN
    description = "Use when you wanna know about user's family."

    def parameters_schema(self) -> dict:
        return RetrievalAgentInput.schema()
