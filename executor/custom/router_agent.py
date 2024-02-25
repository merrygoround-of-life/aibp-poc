from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import Field

from executor.core.functioncall_agent import FunctionCallAgent, FunctionCallConfig
from executor.core.orchestration_agent import OrchestrationAgentInput


class RouterAgentInput(OrchestrationAgentInput):
    local_ip: str = Field(description="user local ip")


class RouterAgent(FunctionCallAgent):
    def __init__(self, config: FunctionCallConfig, tool_agents=None, uuid=None):
        super().__init__(tool_agents=tool_agents, config=config, uuid=uuid)

    def get_additional_messages(self, agent_input: dict) -> list:
        local_ip = RouterAgentInput.model_validate(agent_input).local_ip
        return [ChatCompletionUserMessageParam(
            role="user",
            content=f"My ip is {local_ip}."
        )]
