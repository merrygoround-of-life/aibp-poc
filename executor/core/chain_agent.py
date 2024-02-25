from pydantic import Field, BaseModel

from executor.core.agent import Agent, AgentConfig, AgentType


class AgentChain(BaseModel):
    """
    Model for AgentChain
    """
    import_key: str = Field(description="previous output key name to import")
    to_key: str = Field(description="new key name to replace to", default=None)


class ChainConfig(AgentConfig):
    chain_map: list[list[AgentChain]] = Field(description="chain mapping", default=[])


class ChainAgent(Agent):
    type = AgentType.CHAIN
    description = "Use when you wanna execute sub-agents sequentially."

    def __init__(self, config: ChainConfig, chain_agents: list[Agent], uuid=None):
        super().__init__(config=config, uuid=uuid)
        self.chain_agents: list[Agent] = chain_agents
        self.chain_map: list[list[AgentChain]] = ChainConfig.model_validate(config).chain_map

    def parameters_schema(self) -> dict:
        if self.chain_agents:
            return self.chain_agents[0].parameters_schema()

        return {}

    def invoke(self, user: str, agent_input: dict) -> dict:
        sub_agent_output_context = agent_input
        sub_agent_output = {}

        for sub_agent, sub_agent_chainmap in zip(self.chain_agents, self.chain_map):
            sub_agent_output_context = {**sub_agent_output_context, **sub_agent_output}
            sub_agent_input = {}
            for chain in sub_agent_chainmap:
                to_key = chain.to_key if chain.to_key else chain.import_key
                sub_agent_input[to_key] = sub_agent_output_context[chain.import_key]

            sub_agent_output = sub_agent.execute(user=user, agent_input=sub_agent_input)

        return sub_agent_output
