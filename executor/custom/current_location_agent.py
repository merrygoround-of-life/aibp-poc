import requests
from pydantic import Field, BaseModel

from executor.core.agent import Agent, AgentConfig, AgentInput


class CurrentLocationInput(AgentInput):
    local_ip: str = Field(description="local ip")


class CurrentLocationAgent(Agent):
    description = "Use when you want to know where you're in."

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def parameters_schema(self) -> dict:
        return CurrentLocationInput.schema()

    def invoke(self, user: str, agent_input: dict) -> dict:
        class IpApiResponse(BaseModel):
            city: str

        local_ip = CurrentLocationInput.model_validate(agent_input).local_ip
        ipapi_response = IpApiResponse.model_validate(requests.get(f"http://ip-api.com/json/{local_ip}").json())

        return {"location": ipapi_response.city}
