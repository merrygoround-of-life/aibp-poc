from abc import ABCMeta, abstractmethod
from datetime import datetime
from enum import Enum
from uuid import uuid4

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel


class AgentType(int, Enum):
    TOOL = 0
    CHAIN = 1
    ORCHESTRATION = 2


class AgentInput(BaseModel):
    """
    Model for Agent Input
    """
    pass


class AgentConfig(BaseModel):
    """
    Model for Agent Config
    """
    pass


class Agent(metaclass=ABCMeta):
    type: AgentType = AgentType.TOOL
    description: str

    def __init__(self, config: AgentConfig, uuid=None):
        self.uuid: str = uuid if uuid else str(uuid4())
        self.config: AgentConfig = config

    def tool_schema(self) -> dict:
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.uuid,
                description=self.description,
                parameters=self.parameters_schema()
            )
        )

    def parameters_schema(self) -> dict:
        return AgentInput.schema()

    def execute(self, user: str, agent_input: dict) -> dict:
        print(f"[{self.__class__.__name__}] invoking")
        start = datetime.now()

        output = self.invoke(user, agent_input)

        end = datetime.now()
        print(f"[{self.__class__.__name__}] invoked ({format((end - start).total_seconds() * 1000, '.3f')} ms)")

        return output

    @abstractmethod
    def invoke(self, user: str, agent_input: dict) -> dict:
        pass
