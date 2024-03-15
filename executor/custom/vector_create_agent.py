import numpy as np
from pydantic import Field

from executor.core.agent import AgentInput, AgentType, AgentConfig
from executor.custom.vector_agent import VectorAgent


class VectorCreateAgentInput(AgentInput):
    texts: list[str] = Field(description="array of strings to create")
    vectors: list[list[float]] = Field(description="array of vectors to create")


class VectorCreateAgent(VectorAgent):
    type = AgentType.TOOL
    description = "Use when you wanna create vector db"

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def invoke(self, user: str, agent_input: dict) -> dict:
        texts = VectorCreateAgentInput.model_validate(agent_input).texts
        vectors = VectorCreateAgentInput.model_validate(agent_input).vectors

        size = len(VectorAgent.doc_store)
        for idx, text in enumerate(texts):
            VectorAgent.doc_store[idx + size] = text

        np_vectors = np.array(vectors, dtype=np.float32)
        VectorAgent.index.add(np_vectors)
        return {}
