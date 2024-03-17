import os

from openai import OpenAI
from pydantic import Field

from executor.core.agent import AgentConfig, AgentInput, Agent, AgentType


class EmbeddingAgentInput(AgentInput):
    texts: list[str] = Field(description="array of strings to embed")


class EmbeddingConfig(AgentConfig):
    model: str = Field(description="model to use", default="text-embedding-3-small")


class EmbeddingAgent(Agent):
    type = AgentType.TOOL
    description = "Use when you wanna acquire embeddings of some texts."

    def __init__(self, config=EmbeddingConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)
        self.__model = config.model
        self.__client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def invoke(self, user: str, agent_input: dict) -> dict:
        embedding_input = EmbeddingAgentInput.model_validate(agent_input)

        embedding_response = self.__client.embeddings.create(
            model=self.__model,
            input=embedding_input.texts,
            encoding_format="float"
        )

        embeddings = [embedding.embedding for embedding in embedding_response.data]
        return {"embeddings": embeddings}
