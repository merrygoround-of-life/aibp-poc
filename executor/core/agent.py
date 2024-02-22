from abc import ABCMeta, abstractmethod
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentType(int, Enum):
    TOOL = 0
    CHAIN = 1
    ORCHESTRATION = 2


class AgentInput(BaseModel):
    """
    Model for Agent Input
    """
    pass


class AgentChain(BaseModel):
    """
    Model for Agent AgentChain
    """
    from_key: str = Field(description="previous output key name to transfer from")
    to_key: str = Field(description="current input key name to transfer to")


class AgentConfig(BaseModel):
    """
    Model for Agent Config
    """
    pass


class Agent(metaclass=ABCMeta):
    type: AgentType = AgentType.TOOL
    description: str

    def __init__(self, config: AgentConfig, name: str = None):
        self.name: str = name if name else str(uuid4())
        self.config: AgentConfig = config

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema()
            }
        }

    def parameters_schema(self) -> dict:
        return AgentInput.schema()

    @abstractmethod
    def execute(self, agent_input: dict) -> dict:
        pass
