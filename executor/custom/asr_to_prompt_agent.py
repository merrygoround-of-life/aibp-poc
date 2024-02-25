from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentInput


class AsrInput(AgentInput):
    text: str = Field(description="asr text")


class AsrToPromptInput(AgentInput):
    asr_input: AsrInput = Field(description="asr input")


class AsrToPromptAgent(Agent):
    description = "xxx"

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def invoke(self, user: str, agent_input: dict) -> dict:
        asr_input = AsrToPromptInput.model_validate(agent_input).asr_input

        return {"prompt": asr_input.text}
