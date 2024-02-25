import requests
from pydantic import BaseModel

from executor.core.agent import Agent, AgentConfig


class UserProfileAgent(Agent):
    description = "Use when you want to know your profile."

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def invoke(self, user: str, agent_input: dict) -> dict:
        class MyIpResponse(BaseModel):
            ip: str

        myip_response = MyIpResponse.model_validate(requests.get("https://api.myip.com/").json())

        return {
            "name": "Isaac Park",
            "company": "plantrue.com",
            "ip": myip_response.ip
        }
