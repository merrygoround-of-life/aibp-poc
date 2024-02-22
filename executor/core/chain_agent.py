from pydantic import Field

from executor.core.agent import Agent, AgentConfig, AgentType, AgentChain


class ChainConfig(AgentConfig):
    chain_map: list[list[AgentChain]] = Field(description="chain mapping", default=[])


class ChainAgent(Agent):
    type = AgentType.CHAIN
    description = "Use when you wanna execute sub-agents sequentially."

    def __init__(self, chain_agents: list[Agent], config=ChainConfig(), name: str = None):
        super().__init__(config=config, name=name)
        self.chain_agents: list[Agent] = chain_agents
        self.chain_map: list[list[AgentChain]] = ChainConfig.parse_obj(config).chain_map

    def parameters_schema(self) -> dict:
        if self.chain_agents:
            return self.chain_agents[0].parameters_schema()

        return {}

    def execute(self, agent_input: dict) -> dict:
        sub_agent_output_context = agent_input
        sub_agent_output = {}

        for sub_agent, sub_agent_chainmap in zip(self.chain_agents, self.chain_map):
            sub_agent_output_context = {**sub_agent_output_context, **sub_agent_output}
            sub_agent_input = {}
            for chain in sub_agent_chainmap:
                sub_agent_input[chain.to_key] = sub_agent_output_context[chain.from_key]

            sub_agent_output = sub_agent.execute(sub_agent_input)

        return sub_agent_output
