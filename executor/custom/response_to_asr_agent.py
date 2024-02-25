from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput


class ResponseToAsrInput(AgentInput):
    response: str = Field(description="response")
    company: str = Field(description="company")


class ResponseToAsrAgent(Agent):
    description = "xxx"

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def invoke(self, user: str, agent_input: dict) -> dict:
        response_input = ResponseToAsrInput.model_validate(agent_input)

        return {
            "asr_output": {
                "response": response_input.response,
                "company": response_input.company
            }
        }
