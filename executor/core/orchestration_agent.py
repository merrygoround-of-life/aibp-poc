from abc import abstractmethod

from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput, AgentType
from memory.manager import MemoryManager


class OrchestrationAgentInput(AgentInput):
    prompt: str = Field(description="user prompt")


class OrchestrationAgent(Agent):
    type = AgentType.ORCHESTRATION
    description = "Use when you wanna execute sub-agents orchestrated by llm."

    memory_manager = MemoryManager()

    def __init__(self, config: AgentConfig, tool_agents=None, uuid=None):
        super().__init__(config=config, uuid=uuid)
        self.tool_agents: list[Agent] = tool_agents if tool_agents else []
        self.tool_schemas: list[dict] = [tool_agent.tool_schema() for tool_agent in self.tool_agents]

    @abstractmethod
    def reason(self, messages):
        pass

    def get_additional_messages(self, agent_input: dict) -> list:
        return []
