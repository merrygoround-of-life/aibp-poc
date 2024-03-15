from abc import abstractmethod

import faiss

from executor.core.agent import AgentConfig, Agent, AgentType


class VectorAgent(Agent):
    type = AgentType.TOOL
    description = "Use when you wanna vector db"

    index = faiss.IndexHNSWFlat(1536, 32)
    doc_store: dict[int, str] = {}

    def __init__(self, config=AgentConfig(), uuid=None):
        super().__init__(config=config, uuid=uuid)

    @abstractmethod
    def invoke(self, user: str, agent_input: dict) -> dict:
        pass
