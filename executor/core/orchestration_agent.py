from abc import abstractmethod

from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput, AgentType
from memory.manager import MemoryManager


class OrchestrationAgentInput(AgentInput):
    prompt: str = Field(description="user prompt")
    user: str = Field(description="user id")


class OrchestrationConfig(AgentConfig):
    system_message: str = Field(description="system message", default="You are a helpful assistant.")


class OrchestrationAgent(Agent):
    type = AgentType.ORCHESTRATION
    description = "Use when you wanna execute sub-agents orchestrated by llm."

    memory_manager = MemoryManager()

    def __init__(self, tool_agents: list[Agent]=[], config=OrchestrationConfig(), name: str = None):
        super().__init__(config=config, name=name)
        self.tool_agents: list[Agent] = tool_agents
        self.tool_schemas: list[dict] = [tool_agent.tool_schema() for tool_agent in self.tool_agents]
        self.system_message = OrchestrationConfig.parse_obj(config).system_message

    def parameters_schema(self) -> dict:
        return OrchestrationAgentInput.schema()

    @abstractmethod
    def execute(self, agent_input: dict) -> dict:
        pass
