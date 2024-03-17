import numpy as np
from pydantic import Field

from executor.core.agent import AgentInput, AgentType, AgentConfig
from executor.custom.vector_agent import VectorAgent


class VectorQueryAgentInput(AgentInput):
    vectors: list[list[float]] = Field(description="array of vector to query")


class VectorQueryAgent(VectorAgent):
    type = AgentType.TOOL
    description = "Use when you wanna vector query of some embeddings"

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    def invoke(self, user: str, agent_input: dict) -> dict:
        vectors = VectorQueryAgentInput.model_validate(agent_input).vectors

        np_vectors = np.array(vectors, dtype=np.float32)
        _, indices = VectorAgent.index.search(np_vectors, 1)

        outputs = [index[0] for index in indices.tolist()]
        results = [VectorAgent.doc_store[output] if output != -1 else "No Data" for output in outputs]
        return {"results": results}
